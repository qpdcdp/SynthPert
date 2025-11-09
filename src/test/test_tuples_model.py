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
import os
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset, PerturbationPredictionDataset, MetabolicFluxDataset, SeahorseClusterDataset

# Import Accelerator
from accelerate import Accelerator

# --- Configuration ---
def main(args):

    # ==================== VERIFICATION BLOCK ====================
    # This block MUST be the very first thing in your main function.
    # No other code (except imports) should come before it.
    
    # 1. Read the environment variable set by `accelerate launch`
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"--> [Rank {local_rank}/{world_size}] Starting up. Reading LOCAL_RANK.")
    except KeyError:
        # This will happen if not using `accelerate launch`, for single-GPU debugging
        local_rank = 0
        print("--> [Rank 0] LOCAL_RANK not set. Defaulting to 0.")

    # 2. Set the device for this specific process
    torch.cuda.set_device(local_rank)

    # 3. Print a confirmation from EACH process
    # We expect to see this line from all 8 processes at the very start of the log.
    print(f"--> [Rank {local_rank}] Verification PASSED. Device explicitly set to: {torch.cuda.current_device()}")
    # ==========================================================

    # Set seeds for reproducibility from the start
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Starting tuples evaluation script...")


    
    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = "eval.jsonl" # Define filename separately

    batch_size = args.batch_size # batch size per device
    batches_to_print = 0 # How many batches to print for debugging

    # --- Helper Function to Parse Answer ---
    def extract_answer(generated_text):
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract, strip whitespace, and return the answer directly.
            answer = match.group(1).strip()
            return answer
        return None 

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
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
            )
        model.eval()

    # Now all processes have access to the model/tokenizer objects (though not yet prepared)
    if accelerator.is_main_process:
        print("Model and Tokenizer loaded on main process.")

    # 2. Load Test Dataset
    # Dataset loading needs to happen on all processes.
    if accelerator.is_main_process:
        print("Loading test dataset definition...")
    if args.generate_all_non_de_samples or args.generate_4x_non_de_samples:
        prompt_mode = "warning_different_distributions"
    elif args.gene_enrichment:
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
    elif args.test_mode == "metabolic_flux":
        test_dataset = MetabolicFluxDataset(
            csv_path=csv_data_directory,
            test_split_cell_lines=args.test_split_cell_lines,
            split="test",
        )
    elif args.test_mode == "seahorse_clusters":
        test_dataset = SeahorseClusterDataset(
            csv_path=csv_data_directory,
            test_split_cell_lines=args.test_split_cell_lines,
            split="test",
        )
    else:
        test_dataset = DiffExpressionDataset(
            csv_dir=csv_data_directory, 
            prompt_mode=prompt_mode, 
            test_split_cell_lines= args.test_split_cell_lines, 
            split="test", generate_all_non_de_samples=args.generate_all_non_de_samples, 
            generate_4x_non_de_samples=args.generate_4x_non_de_samples
        )

    # Use a DistributedSampler for fair distribution across processes.
    # This is crucial if `drop_last=True` is not sufficient or if the dataset is large.
    # The Subset needs to be applied *before* the DistributedSampler if you want to sample from a subset of the full data.
    
    # First, apply the fraction to the base dataset
    total_samples = len(test_dataset)
    sample_size = int(total_samples * args.dataset_fraction)  # Use the dataset_fraction argument
    
    # Ensure sample_size is at least num_processes to avoid errors with DistributedSampler
    if sample_size < accelerator.num_processes:
        accelerator.print(f"Warning: sample_size ({sample_size}) is less than num_processes ({accelerator.num_processes}). Adjusting to num_processes.")
        sample_size = accelerator.num_processes

    indices = random.sample(range(total_samples), sample_size)
    test_dataset_subset = torch.utils.data.Subset(test_dataset, indices) # Apply subset here
    
    # Create DistributedSampler for the subset
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset_subset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False, # We want reproducible order for evaluation
        drop_last=True # Ensure all processes get same number of full batches
    )
    
    if accelerator.is_main_process:
         print(f"Full test dataset size: {len(test_dataset)} samples.")
         print(f"Subsampled dataset size: {len(test_dataset_subset)} samples for evaluation (fraction={args.dataset_fraction}).")


    # --- Collate function (Corrected) ---
    def collate_fn(batch):
        # Ensure your collate_fn is robust to empty batches as a safety measure,
        # though drop_last=True in DataLoader + DistributedSampler should prevent it.
        if not batch: # Should not happen with drop_last=True and DistributedSampler
            accelerator.print(f"Rank {accelerator.process_index}: Collated batch received an empty list. This should not happen with DistributedSampler and drop_last=True. Returning None.")
            return None 
        
        prompts = [item.get('prompt', None) for item in batch] # Expects 'prompt' key now
        solutions = [item.get('solution', None) for item in batch]
        perts = [item.get('pert', None) for item in batch]
        genes = [item.get('gene', None) for item in batch]
        cell_types = [item.get('cell_type', None) for item in batch]

        # Filter out None prompts if any occurred
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        if len(valid_indices) != len(prompts):
             warnings.warn(f"Rank {accelerator.process_index}: Some items in batch had missing 'prompt' key. Filtering {len(prompts) - len(valid_indices)} items.")
             # Filter other lists accordingly
             prompts = [prompts[i] for i in valid_indices]
             solutions = [solutions[i] for i in valid_indices]
             perts = [perts[i] for i in valid_indices]
             genes = [genes[i] for i in valid_indices]
             cell_types = [cell_types[i] for i in valid_indices]

        if not prompts: # If batch becomes empty after filtering
             accelerator.print(f"Rank {accelerator.process_index}: Collated batch became empty after filtering prompts. Returning None.")
             return None

        try:
            tokenized_output = tokenizer.apply_chat_template(
                prompts, padding=True, return_tensors="pt", add_generation_prompt=True
            )
        except Exception as e:
             accelerator.print(f"Rank {accelerator.process_index}: Error during apply_chat_template: {e}")
             accelerator.print(f"Rank {accelerator.process_index}: Problematic prompts snippet: {prompts[:2]}")
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
            "original_prompts": prompts, # Keep original prompts for context if needed
            "perts": perts,             # <<< Pass metadata through
            "genes": genes,             # <<< Pass metadata through
            "cell_types": cell_types,   # <<< Pass metadata through
        }

    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset_subset, # Use the subset here
        batch_size=batch_size,
        sampler=test_sampler, # Use the DistributedSampler
        collate_fn=collate_fn,
        num_workers=0,
        # shuffle=False is implicitly handled by DistributedSampler for evaluation
        # drop_last=True is already handled by DistributedSampler for evaluation
    )
    
    # Prepare model, dataloader, and sampler
    # This prepares the model for DDP and makes the dataloader distributed-ready,
    # also distributing the sampler for correct index generation per rank.
    model, test_dataloader, test_sampler = accelerator.prepare(model, test_dataloader, test_sampler)

    # ==================== CRITICAL DEBUGGING LINE (UPDATED) ====================
    # Print this AFTER accelerator.prepare to ensure all processes successfully reached this point.
    accelerator.print(f"[Process {accelerator.process_index}] Successfully prepared. My dataloader has {len(test_dataloader)} batches.")
    # ===========================================================================


    # 3. Run Inference and Evaluate
    total_correct_local = 0
    total_evaluated_local = 0
    results_local = []
    samples_printed_count = 0

    accelerator.print(f"\nStarting distributed evaluation...")
    # Use accelerator.is_local_main_process for tqdm on only one rank per node.
    # If you want tqdm on only ONE process globally, use accelerator.is_main_process.
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, file=sys.stdout)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # ==================== ADDED PROGRESS BAR UPDATE ====================
            # This will update the tqdm description with current batch number for local main process.
            # Other processes will log this to their specific output.
            progress_bar.set_description(f"Rank {accelerator.process_index} Evaluating | Batch {batch_idx+1}/{len(test_dataloader)}")
            # ===================================================================

            if batch is None: # Skip batch if collate_fn returned None
                 accelerator.print(f"Skipping empty or problematic batch {batch_idx} on Rank {accelerator.process_index}")
                 continue

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            solutions = batch['solutions']
            # Retrieve metadata passed from collate_fn
            original_prompts = batch['original_prompts']
            perts = batch['perts']
            genes = batch['genes']
            cell_types = batch['cell_types']

            unwrapped_model = accelerator.unwrap_model(model)
            try:
                outputs = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=1,
                )
            except Exception as e:
                accelerator.print(f"Error during model.generate on Rank {accelerator.process_index}, Batch {batch_idx}: {e}")
                # Log the batch content if it fails, for debugging
                accelerator.print(f"Problematic input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
                accelerator.print(f"Problematic original_prompts (first 2): {original_prompts[:2]}")
                # Optionally skip to next batch or handle error
                continue # Skip batch on generation error


            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Compare generated answers with solutions locally
            for i, gen_text in enumerate(generated_texts):
                extracted_answer = extract_answer(gen_text)
                correct_solution = solutions[i]
                is_correct = (extracted_answer.lower() == correct_solution.lower()) if extracted_answer else False

                if is_correct:
                    total_correct_local += 1
                total_evaluated_local += 1

                # Append results including metadata
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
                    "is_correct": is_correct
                })

    # 4. Gather Results Across All Processes
    progress_bar.close() 

    # --- CHANGE 2: Implement File-Based Aggregation ---
    # Each process saves its own results to a uniquely named file.
    temp_results_file = output_dir / f"temp_results_rank_{accelerator.process_index}.jsonl"
    with open(temp_results_file, 'w', encoding='utf-8') as f:
        for item in results_local:
            f.write(json.dumps(item) + '\n')
    
    accelerator.print(f"Rank {accelerator.process_index} finished and saved temporary results to {temp_results_file}")

    # Use a barrier to wait for all processes to finish writing their files.
    accelerator.wait_for_everyone()

    # --- CHANGE 3: The rest of the script now only runs on the main process ---
    if accelerator.is_main_process:
        print("\n--- Aggregating results from all processes ---")
        all_results = []
        # Loop through the temporary files from all ranks and combine them
        for i in range(accelerator.num_processes):
            rank_file = output_dir / f"temp_results_rank_{i}.jsonl"
            if rank_file.exists():
                with open(rank_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line))
                rank_file.unlink() # Clean up the temporary file
            else:
                print(f"Warning: Could not find temporary file for rank {i}: {rank_file}")
        
        # --- Save Combined Results ---
        if all_results:
            print(f"Processed {len(all_results)} detailed results for saving.")
            output_path = output_dir / output_filename
            print(f"Saving combined JSONL results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for result_item in all_results:
                    f.write(json.dumps(result_item) + '\n')
            print("JSONL Results saved successfully.")
        else:
            print("No detailed results were gathered. Skipping JSONL save and detailed stats.")

        # --- Calculate and Save Detailed Statistics (No changes needed below this line) ---
        if all_results:
            stats_string_parts = []
            
            total_evaluated = len(all_results)
            total_correct = sum(1 for item in all_results if item['is_correct'])
            overall_accuracy_percentage = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0

            stats_string_parts.append("--- Overall Evaluation Summary ---")
            stats_string_parts.append(f"Total Samples Evaluated: {total_evaluated}")
            stats_string_parts.append(f"Correct Predictions: {total_correct}")
            stats_string_parts.append(f"Overall Accuracy: {overall_accuracy_percentage:.2f}%\n")
            
            y_true = [item.get('correct_solution') for item in all_results]
            y_pred_raw = [item.get('extracted_answer') for item in all_results]
            
            if args.test_mode == "metabolic_flux":
                defined_classes = ["increased", "decreased", "not changed"]
            elif args.test_mode == "seahorse_clusters":
                defined_classes = ['Increased oxygen consumption rate to extracellular acidification rate ratio', 'Increased extracellular acidification rate and ATP-linked respiration', 'Increased Maximal Respiration', 'Increased Proton Leak', 'Loss of oxidative metabolism', 'No Change']
            elif args.test_mode == "inverse_problem":
                defined_classes = all_results['pert'].unique().tolist()
            else:
                defined_classes = ["upregulated", "downregulated", "not differentially expressed"]


            report_labels = defined_classes + ["none_extracted"]

            y_pred_for_report = [pred if pred in defined_classes else "none_extracted" for pred in y_pred_raw]

            classification_report_path = output_dir / "classification_report.txt"
            class_report_str = "  Not enough data to generate classification report."
            if y_true and y_pred_for_report:
                class_report_str = classification_report(
                    y_true, y_pred_for_report,
                    labels=report_labels, target_names=report_labels,
                    zero_division=0, digits=2
                )

            with open(classification_report_path, 'w', encoding='utf-8') as f:
                f.write(class_report_str)
            print(f"Classification report saved to {classification_report_path}")
            prediction_distribution = {label: 0 for label in report_labels}
            prediction_distribution["other"] = 0
            stats_string_parts.append("--- Classification Report ---")
            stats_string_parts.append(class_report_str + "\n")
            stats_string_parts.append("--- Predicted Answer Distribution (Model Output) ---")
            if total_evaluated > 0:
                for pred in y_pred_raw:
                    if pred in prediction_distribution:
                        prediction_distribution[pred] += 1
                    else:
                        prediction_distribution["other"] +=1

                for answer_type, count in sorted(prediction_distribution.items()):
                    percentage = (count / total_evaluated) * 100
                    stats_string_parts.append(f"  Predicted as '{answer_type}': {count} times ({percentage:.2f}%)")
            else:
                stats_string_parts.append("  No samples evaluated for distribution.")
            stats_string_parts.append("\n")

            # 4. Accuracy by Cell Type
            stats_string_parts.append("--- Accuracy by Cell Type ---")
            cell_type_stats = {}
            for item in all_results:
                cell_type = item.get('cell_type', 'Unknown_Cell_Type')
                if cell_type not in cell_type_stats: cell_type_stats[cell_type] = {'correct': 0, 'total': 0}
                cell_type_stats[cell_type]['total'] += 1
                if item.get('is_correct'): cell_type_stats[cell_type]['correct'] += 1

            if cell_type_stats:
                for cell_type_key in sorted(cell_type_stats.keys()):
                    data = cell_type_stats[cell_type_key]
                    accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
                    stats_string_parts.append(f"  {cell_type_key}: {accuracy:.2f}% ({data['correct']}/{data['total']})")
            else:
                stats_string_parts.append("  No cell type data available.")
            stats_string_parts.append("\n")

            # 5. Accuracy by Perturbation
            stats_string_parts.append("--- Accuracy by Perturbation ---")
            pert_stats = {}
            for item in all_results:
                pert = item.get('pert', 'Unknown_Perturbation')
                if pert not in pert_stats: pert_stats[pert] = {'correct': 0, 'total': 0}
                pert_stats[pert]['total'] += 1
                if item.get('is_correct'): pert_stats[pert]['correct'] += 1

            if pert_stats:
                for pert_key in sorted(pert_stats.keys()):
                    data = pert_stats[pert_key]
                    accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
                    stats_string_parts.append(f"  {pert_key}: {accuracy:.2f}% ({data['correct']}/{data['total']})")
            else:
                stats_string_parts.append("  No perturbation data available.")
            stats_string_parts.append("\n")

            # 6. Save the combined statistics
            stats_filename = "evaluation_statistics.txt"
            stats_output_path = output_dir / stats_filename
            with open(stats_output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(stats_string_parts))
            print(f"Detailed statistics saved to {stats_output_path}")

    # Ensure all processes finish cleanly
    accelerator.wait_for_everyone()
    print("Script finished successfully on all processes.")