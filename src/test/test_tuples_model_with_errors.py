from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
from pathlib import Path
import json
import warnings
import re
import torch
import sys
import random
import numpy as np # Import numpy for statistical calculations
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report
# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset

from torch.utils.data import Dataset
# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

class FoldDatasetWrapper(Dataset):
    """
    Wraps an existing dataset to assign a fold_id to each item.
    The dataset is shuffled once internally to ensure folds are random.
    """
    def __init__(self, original_dataset, num_folds):
        self.original_dataset = original_dataset
        self.num_folds = num_folds
        
        # Shuffle indices once at the beginning
        self.shuffled_indices = list(range(len(original_dataset)))
        random.seed(42) # Use a fixed seed for reproducibility
        random.shuffle(self.shuffled_indices)

        # Create a map from the original index to its fold ID
        self.index_to_fold_id = [0] * len(original_dataset)
        fold_size = len(original_dataset) // num_folds
        for i, original_idx in enumerate(self.shuffled_indices):
            # Assign fold ID based on the position in the shuffled list
            fold_id = min(i // fold_size, num_folds - 1)
            self.index_to_fold_id[original_idx] = fold_id

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get the original data item
        data_item = self.original_dataset[idx]
        
        # Add the pre-calculated fold_id
        # This works even if the DataLoader shuffles again, because `idx` is the original index
        data_item['fold_id'] = self.index_to_fold_id[idx]
        return data_item

# --- Configuration ---
def main(args):
    print("Starting tuples evaluation script...")


    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = "sft_lora.jsonl" # Define filename separately

    batch_size = args.batch_size # batch size per device
    batches_to_print = 0 # How many batches to print for debugging

    # --- Helper Function to Parse Answer ---
    def extract_answer(generated_text):
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip().lower()
            if answer in ["upregulated", "downregulated", "not differentially expressed"]: return answer
            else:
                if "upregulated" in answer: return "upregulated"
                if "downregulated" in answer: return "downregulated"
                if "not differentially expressed" in answer: return "not differentially expressed"
                # If a partial match is found, return the canonical version
                return "none_extracted" # Return a specific token for no valid answer
        return "none_extracted"


    # --- Initialize Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Process {accelerator.process_index} using device: {device}")

    # --- Load Model and Tokenizer (on main process first) ---
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            if accelerator.is_main_process:
                print("Warning: Tokenizer missing pad token; setting to eos_token.")
            tokenizer.pad_token = xxxx
        tokenizer.padding_side = "left"
        
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
    
        model.eval()

    if accelerator.is_main_process:
        print("Model and Tokenizer loaded on main process.")
    
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            print("Loading and subsampling test dataset on main process...")

        if args.generate_all_non_de_samples or args.generate_4x_non_de_samples:
            prompt_mode = "warning_different_distributions"
        elif args.gene_enrichment:
            prompt_mode = "gene_enrichment"
        else:
            prompt_mode = "default"
            
        # Step 1: Load the full dataset definition. This is lightweight.
        full_test_dataset = DiffExpressionDataset(
            csv_dir=csv_data_directory, prompt_mode=prompt_mode, 
            test_split_cell_lines=args.test_split_cell_lines, split="test", 
            generate_all_non_de_samples=args.generate_all_non_de_samples, 
            generate_4x_non_de_samples=args.generate_4x_non_de_samples, 
        )
        
        # Step 2: Perform random sampling ONLY on the main process.
        if args.dataset_fraction < 1.0:
            total_samples = len(full_test_dataset)
            sample_size = int(total_samples * args.dataset_fraction)
            
            random.seed(42) 
            indices = random.sample(range(total_samples), sample_size)
            test_dataset = torch.utils.data.Subset(full_test_dataset, indices)
        else:
            test_dataset = full_test_dataset

        # Step 3: Wrap for k-folding if needed.
        if args.num_folds > 1:
            accelerator.print(f"Wrapping dataset and assigning samples to {args.num_folds} folds.")
            test_dataset = FoldDatasetWrapper(test_dataset, args.num_folds)

    if accelerator.is_main_process:
         print(f"Full test dataset size: {len(test_dataset)} samples.")

    # --- Collate function (Corrected and Updated) ---
    def collate_fn(batch):
        # Extract necessary fields, use .get for safety with metadata
        prompts = [item.get('prompt', None) for item in batch] # Expects 'prompt' key now
        solutions = [item.get('solution', None) for item in batch]
        perts = [item.get('pert', None) for item in batch]
        genes = [item.get('gene', None) for item in batch]
        cell_types = [item.get('cell_type', None) for item in batch]
        fold_ids = [item.get('fold_id', -1) for item in batch] # Default to -1 if not present

        # Filter out None prompts if any occurred
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        if len(valid_indices) != len(prompts):
             warnings.warn("Some items in batch had missing 'prompt' key.")
             # Filter other lists accordingly
             prompts = [prompts[i] for i in valid_indices]
             solutions = [solutions[i] for i in valid_indices]
             perts = [perts[i] for i in valid_indices]
             genes = [genes[i] for i in valid_indices]
             cell_types = [cell_types[i] for i in valid_indices]
             fold_ids = [fold_ids[i] for i in valid_indices] # <<< Make sure to filter fold_ids too

        if not prompts: # If batch becomes empty after filtering
             return None

        try:
            tokenized_output = tokenizer.apply_chat_template(
                prompts, padding=True, return_tensors="pt", add_generation_prompt=True
            )
        except Exception as e:
             accelerator.print(f"Error during apply_chat_template: {e}")
             accelerator.print(f"Problematic prompts snippet: {prompts[:2]}")
             # Return None or raise error to stop processing this batch
             return None # Skip this batch


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
            "fold_ids": fold_ids, # <<< Pass fold_ids through
        }

    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Prepare model and dataloader
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # 3. Run Inference and Evaluate
    results_local = []
    
    accelerator.print(f"\nStarting distributed evaluation...")
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, file=sys.stdout)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                 accelerator.print(f"Skipping empty or problematic batch {batch_idx} on Rank {accelerator.process_index}")
                 continue

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            solutions = batch['solutions']
            original_prompts = batch['original_prompts']
            perts = batch['perts']
            genes = batch['genes']
            cell_types = batch['cell_types']
            fold_ids = batch['fold_ids'] # <<< Retrieve fold_ids

            unwrapped_model = accelerator.unwrap_model(model)
            try:
                outputs = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False, # Using do_sample=False for more deterministic output
                )
            except Exception as e:
                accelerator.print(f"Error during model.generate on Rank {accelerator.process_index}, Batch {batch_idx}: {e}")
                continue


            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, gen_text in enumerate(generated_texts):
                extracted_answer = extract_answer(gen_text)
                correct_solution = solutions[i]
                is_correct = (extracted_answer == correct_solution)

                user_prompt_content = ""
                if original_prompts[i] and isinstance(original_prompts[i], list) and len(original_prompts[i]) > 1:
                   user_prompt_content = original_prompts[i][-1].get('content', "")

                results_local.append({
                    "user_prompt": user_prompt_content,
                    "pert": perts[i],
                    "gene": genes[i],
                    "cell_type": cell_types[i],
                    "generated_text": gen_text,
                    "extracted_answer": extracted_answer,
                    "correct_solution": correct_solution,
                    "is_correct": is_correct,
                    "fold_id": fold_ids[i], # <<< Store fold_id with each result
                })

    # 4. Gather Results Across All Processes
    progress_bar.close() 
    temp_results_file = output_dir / f"temp_results_rank_{accelerator.process_index}.jsonl"
    with open(temp_results_file, 'w', encoding='utf-8') as f:
        for item in results_local:
            f.write(json.dumps(item) + '\n')
            
    accelerator.print(f"Rank {accelerator.process_index} finished and saved temporary results.")
    accelerator.wait_for_everyone()

    # --- AGGREGATION AND ANALYSIS (MAIN PROCESS ONLY) ---
    if accelerator.is_main_process:
        print("\n--- Aggregating results from all processes ---")
        all_results = []
        for i in range(accelerator.num_processes):
            rank_file = output_dir / f"temp_results_rank_{i}.jsonl"
            if rank_file.exists():
                with open(rank_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line))
                rank_file.unlink()
            else:
                print(f"Warning: Could not find temporary file for rank {i}: {rank_file}")

        # --- START OF NEW, DETAILED ANALYSIS ---
        stats_string_parts = ["--- k-Fold Evaluation Summary ---"]
        
        # 1. Overall Pooled Accuracy
        total_correct = sum(1 for item in all_results if item['is_correct'])
        total_evaluated = len(all_results)
        pooled_accuracy = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
        stats_string_parts.append(f"Overall Pooled Accuracy: {pooled_accuracy:.2f}% ({total_correct}/{total_evaluated})\n")

        # 2. K-Fold Detailed Analysis
        if args.num_folds > 1 and total_evaluated > 0:
            stats_string_parts.append(f"--- Statistics Based on {args.num_folds}-Fold Splitting ---")

            # Initialize structures to hold metrics from each fold
            fold_accuracies = []
            defined_classes = ["upregulated", "downregulated", "not differentially expressed"]
            # A dictionary to hold lists of metrics for each class
            per_class_metrics = {cls: {'precision': [], 'recall': [], 'f1-score': []} for cls in defined_classes}

            # Calculate metrics for each fold
            for fold_id in range(args.num_folds):
                fold_results = [r for r in all_results if r.get('fold_id') == fold_id]
                if not fold_results:
                    stats_string_parts.append(f"  Fold {fold_id+1}/{args.num_folds}: No results found.")
                    continue

                y_true = [r['correct_solution'] for r in fold_results]
                y_pred = [r['extracted_answer'] for r in fold_results]
                
                # Calculate fold accuracy
                fold_correct = sum(1 for r in fold_results if r['is_correct'])
                fold_total = len(fold_results)
                fold_acc = (fold_correct / fold_total) * 100 if fold_total > 0 else 0
                fold_accuracies.append(fold_acc)

                # Generate classification report as a dictionary
                report = classification_report(y_true, y_pred, labels=defined_classes, output_dict=True, zero_division=0)

                # Store precision, recall, f1 for each class
                for cls in defined_classes:
                    if cls in report:
                        per_class_metrics[cls]['precision'].append(report[cls]['precision'])
                        per_class_metrics[cls]['recall'].append(report[cls]['recall'])
                        per_class_metrics[cls]['f1-score'].append(report[cls]['f1-score'])
                    else: # Handle case where a class has no instances in a fold
                        per_class_metrics[cls]['precision'].append(0)
                        per_class_metrics[cls]['recall'].append(0)
                        per_class_metrics[cls]['f1-score'].append(0)

            # 3. Calculate and Format Summary Statistics
            if fold_accuracies:
                mean_acc = np.mean(fold_accuracies)
                std_acc = np.std(fold_accuracies)
                stats_string_parts.append(f"Mean Accuracy: {mean_acc:.2f}% (± {std_acc:.2f})\n")
            
            stats_string_parts.append("--- Per-Class Metrics (Mean ± Std Dev) ---")
            for cls in defined_classes:
                stats_string_parts.append(f"Class: {cls}")
                
                mean_p = np.mean(per_class_metrics[cls]['precision']) * 100
                std_p = np.std(per_class_metrics[cls]['precision']) * 100
                stats_string_parts.append(f"  - Precision: {mean_p:.2f}% (± {std_p:.2f})")

                mean_r = np.mean(per_class_metrics[cls]['recall']) * 100
                std_r = np.std(per_class_metrics[cls]['recall']) * 100
                stats_string_parts.append(f"  - Recall:    {mean_r:.2f}% (± {std_r:.2f})")

                mean_f1 = np.mean(per_class_metrics[cls]['f1-score']) * 100
                std_f1 = np.std(per_class_metrics[cls]['f1-score']) * 100
                stats_string_parts.append(f"  - F1-Score:  {mean_f1:.2f}% (± {std_f1:.2f})\n")
        
        else:
            stats_string_parts.append("\n(num_folds=1, detailed cross-validation statistics not applicable)")

        # Save the final, comprehensive summary
        stats_output_path = output_dir / "evaluation_summary_kfold.txt"
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(stats_string_parts))
        print(f"\nFinal summary saved to {stats_output_path}")

    accelerator.wait_for_everyone()
    print("Script finished successfully on all processes.")