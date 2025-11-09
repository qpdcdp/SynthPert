import time
import argparse
from pathlib import Path
import json
import warnings
import re
import torch
import sys
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
import torch.distributed as dist

import lightning as L
from lightning.fabric import Fabric
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp.wrap import always_wrap_policy
from lightning.fabric.strategies import FSDPStrategy

from src.utils import extract_answer
from src.data import DiffExpressionDataset, PerturbationPredictionDataset


def main(args):
    # Set up paths and configurations
    random.seed(args.seed)
    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_filename = "eval.jsonl"

    # Initialize Fabric
    fsdp_strategy = FSDPStrategy(
        auto_wrap_policy=always_wrap_policy,
        cpu_offload=True # Offloads parameters to CPU during sharding
    )
    fabric = Fabric(
        accelerator="gpu",
        precision="bf16-mixed",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=fsdp_strategy #args.strategy if args.strategy else "auto",
    )
    
    fabric.launch()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = xxxx
    tokenizer.padding_side = "left"

    # Load model
    if args.lora_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )
        from peft import PeftModel
        # Update this path to the correct checkpoint directory  # Using the correct directory
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )

    model = fabric.setup_module(model)
    model.mark_forward_method('generate')

    model.eval()

    if args.gene_enrichment:
        prompt_mode = "gene_enrichment"
    else:
        prompt_mode = "default"

    if args.test_mode == "inverse_problem":
        test_dataset = PerturbationPredictionDataset(
            csv_dir=csv_data_directory,
            prompt_mode=prompt_mode,
            test_split_cell_lines=args.test_split_cell_lines,
            split="test"
        )
    else:
        test_dataset = DiffExpressionDataset(
            csv_dir=csv_data_directory, 
            prompt_mode=prompt_mode, 
            test_split_cell_lines= args.test_split_cell_lines, 
            split="test",
            generate_all_non_de_samples=args.generate_all_non_de_samples, 
            generate_4x_non_de_samples=args.generate_4x_non_de_samples
        )
    print(f"Full test dataset size: {len(test_dataset)} samples.")
    if args.dataset_fraction < 1.0:
        total_samples = len(test_dataset)
        sample_size = int(total_samples * args.dataset_fraction)
        random.seed(42)
        indices = random.sample(range(total_samples), sample_size)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Test dataset size after sampling: {len(test_dataset)} samples.")

    # --- Collate function ---
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch] 
        solutions = [item['solution'] for item in batch]
        perts = [item['pert'] for item in batch]
        genes = [item['gene'] for item in batch]
        cell_types = [item['cell_type'] for item in batch]

        tokenized_output = tokenizer.apply_chat_template(
            prompts, padding=True, return_tensors="pt", add_generation_prompt=True
        )

        if isinstance(tokenized_output, torch.Tensor):
            input_ids_tensor = tokenized_output
            attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
        elif isinstance(tokenized_output, dict) or hasattr(tokenized_output, 'keys'):
            input_ids_tensor = tokenized_output.get('input_ids')
            attention_mask_tensor = tokenized_output.get('attention_mask')
            if input_ids_tensor is None: raise ValueError("Missing 'input_ids'")
            if attention_mask_tensor is None: attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
        else: raise TypeError(f"Unexpected output type from tokenizer: {type(tokenized_output)}")

        # Return all necessary data
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "solutions": solutions,
            "original_prompts": prompts,
            "perts": perts,
            "genes": genes,
            "cell_types": cell_types,
        }

    # Create and setup DataLoader
    dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=4 * args.devices,  # Adjust number of workers based on devices
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    
    # eval variables
    total_correct = 0
    total_evaluated = 0
    results = []
   
    progress_bar = tqdm(dataloader, desc="Evaluating", disable=not fabric.is_global_zero)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            solutions = batch['solutions']

             # Retrieve metadata passed from collate_fn
            original_prompts = batch['original_prompts']
            perts = batch['perts']
            genes = batch['genes']
            cell_types = batch['cell_types']

            with fabric.no_backward_sync(model):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, generated_text in enumerate(generated_texts):
                extracted_answer = extract_answer(generated_text)
                if extracted_answer is None:
                    print(f"Warning: No valid answer extracted from generated text for prompt {original_prompts[i]}")
                    continue
                
                correct_solution = solutions[i]
                is_correct = (extracted_answer == correct_solution)

                if is_correct:
                    total_correct += 1
                total_evaluated += 1

                results.append({
                    "user_prompt": original_prompts[i][-1].get('content', ""), # Last part of the prompt
                    "pert": perts[i],               
                    "gene": genes[i],               
                    "cell_type": cell_types[i],     
                    "generated_text": generated_text,
                    "extracted_answer": extracted_answer,
                    "correct_solution": correct_solution,
                    "is_correct": is_correct
                })
                
            progress_bar.set_postfix({
                "correct": total_correct,
                "evaluated": total_evaluated,
                "accuracy": total_correct / total_evaluated if total_evaluated > 0 else 0
            })
    
    fabric.barrier()
    #gather results across all processes
    all_results = fabric.all_gather(results)

    if fabric.is_global_zero:
        # Flatten the list of results
        flat_results = [item for sublist in all_results for item in sublist]
        
        # Save results to JSONL file
        output_file = output_dir / output_json_filename
        with open(output_file, 'w') as f:
            for result in flat_results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Results saved to {output_file}")
        
        # Print classification report
        y_true = [result['correct_solution'] for result in flat_results]
        y_pred = [result['extracted_answer'] for result in flat_results]
        print("Classification Report:")
        # save classification report to a file
        with open(output_dir / "classification_report.txt", "w") as report_file:
            report_file.write(classification_report(y_true, y_pred, zero_division=0))