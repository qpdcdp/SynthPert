import time
import argparse
from pathlib import Path
import json
import warnings
import torch
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from vllm import LLM, SamplingParams


def main(args):
    # Set up paths and configurations
    random.seed(args.seed)
    np.random.seed(args.seed)    
    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_filename = "eval.jsonl"

    # Load tokenizer


    # --- Model Loading ---
    # vLLM loads the model. If a LoRA checkpoint is provided, we first
    # merge it with the base model and save it to a new directory.
    model_path_for_vllm = args.model_name_or_path

    # if args.lora_checkpoint:
    #     print("Loading base model for LoRA merging...")
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name_or_path,
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True,
    #     )
    #     print("Loading and merging LoRA adapter...")
    #     lora_model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
    #     model = lora_model.merge_and_unload()
        
    #     merged_model_dir = output_dir / "merged_model_for_vllm"
    #     print(f"Saving merged model to {merged_model_dir}...")
    #     model.save_pretrained(merged_model_dir)
    #     tokenizer.save_pretrained(merged_model_dir)
        
    #     model_path_for_vllm = str(merged_model_dir)
    #     print("Model merging and saving complete.")

    # Initialize vLLM
    # The number of GPUs is used for tensor parallelism.
    # vLLM handles model distribution automatically.
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path_for_vllm,
        tensor_parallel_size=args.devices,
        trust_remote_code=True,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_num_seqs=args.vllm_max_num_seqs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = xxxx
    # vLLM handles padding internally, so padding_side is not strictly necessary
    # for the generation process itself but can be good practice.
    tokenizer.padding_side = "left"
    # --- Dataset and Prompt Preparation ---
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
            test_split_cell_lines=args.test_split_cell_lines,
            split="test",
            generate_all_non_de_samples=args.generate_all_non_de_samples,
            generate_4x_non_de_samples=args.generate_4x_non_de_samples
        )
    print(f"Full test dataset size: {len(test_dataset)} samples.")

    if args.dataset_fraction < 1.0:
        total_samples = len(test_dataset)
        sample_size = int(total_samples * args.dataset_fraction)
        indices = random.sample(range(total_samples), sample_size)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Test dataset size after sampling: {len(test_dataset)} samples.")

    # Prepare prompts and store original data for later evaluation
    print("Preparing prompts for vLLM...")
    prompts = [item['prompt'] for item in test_dataset]
    original_items = list(test_dataset) # Keep track of original data
    
    # Use the tokenizer to apply the chat template, creating the final strings for vLLM
    formatted_prompts = tokenizer.apply_chat_template(
        [p for p in prompts],
        tokenize=False,
        add_generation_prompt=True
    )

    # --- vLLM Inference ---
    print("Starting generation with vLLM...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    
    start_time = time.time()
    vllm_outputs = llm.generate(formatted_prompts, sampling_params)
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.2f} seconds.")

    # --- Process and Evaluate Results ---
    results = []
    total_correct = 0
    total_evaluated = 0

    print("Evaluating generated responses...")
    for i, output in enumerate(tqdm(vllm_outputs, desc="Evaluating")):
        original_item = original_items[i]
        generated_text = output.outputs[0].text.strip()

        if i < 5:               # TEMPORARY DEBUGGING CODE: Print the first 5 raw outputs
            print("-" * 50)
            print(f"PROMPT:\n{output.prompt}\n")
            print(f"RAW GENERATED TEXT:\n'{generated_text}'\n")

        extracted_answer = extract_answer(generated_text)
        if extracted_answer is None:
            print(f"Warning: No valid answer extracted from generated text for prompt: {output.prompt}")
            continue

        correct_solution = original_item['solution']
        is_correct = (extracted_answer == correct_solution)

        if is_correct:
            total_correct += 1
        total_evaluated += 1


        results.append({
            "user_prompt": original_item['prompt'][-1].get('content', ""),
            "pert": original_item['pert'],
            "gene": original_item['gene'],
            "cell_type": original_item['cell_type'],
            "generated_text": generated_text,
            "extracted_answer": extracted_answer,
            "correct_solution": correct_solution,
            "is_correct": is_correct
        })

    # --- Save and Report Results ---
    if total_evaluated > 0:
        accuracy = total_correct / total_evaluated
        print(f"\nFinal Accuracy: {accuracy:.4f} ({total_correct}/{total_evaluated})")
    else:
        print("\nNo samples were evaluated.")

    # Save results to a JSONL file
    output_file = output_dir / output_json_filename
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_file}")

    # Generate and save a classification report
    if total_evaluated > 0:
        y_true = [result['correct_solution'] for result in results]
        y_pred = [result['extracted_answer'] for result in results]
        
        print("\nClassification Report:")
        report_str = classification_report(y_true, y_pred, zero_division=0)
        print(report_str)
        
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w") as report_file:
            report_file.write(report_str)
        print(f"Classification report saved to {report_path}")
