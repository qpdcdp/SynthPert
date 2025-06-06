# --- Imports ---
import os # For environment variables
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

import time
import argparse
import sys
from pathlib import Path
import json
import warnings
import re
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch # Import torch for tensor operations

# --- Langchain/OpenAI Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import copy
import random
# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset, GeneRegulationListDataset

# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

# --- Configuration ---
def main(args):


    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_filename = "sft_lora_long.jsonl" # Define filename separately
    # --- BEGIN FIX ---
    output_path = output_dir / (output_filename + "long")# DEFINE output_path HERE
    # --- END FIX ---



    batch_size = 32 # batch size per device
    batches_to_print = 10 # How many batches to print for debugging

    # --- Helper Function to Parse Answer ---
    def extract_answer(generated_text):
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip().lower()
            print(f"DEBUG: Extracted answer: {answer}")  # Debug print
            if answer in ["upregulated", "downregulated", "not differentially expressed"]: return answer
            else:
                if "upregulated" in answer: return "upregulated"
                if "downregulated" in answer: return "downregulated"
                if "not differentially expressed" in answer: return "not differentially expressed"
                return None
        return None

    # --- NEW Helper Functions for AUROC ---
    if args.AUROC_stage == "dir":
        print (f"AUROC stage is 'dir'. Using direction_test_prompt.")

        def map_solution_to_trinary_label(solution_str):
            if solution_str == "upregulated":
                return 1  # Differentially Expressed
            elif solution_str == "downregulated":
                return 0  # Not Differentially Expressed
            elif solution_str == "not differentially expressed":
                return 2
            return -1 # Should not happen if data is clean

        def map_prediction_to_score(extracted_answer_str):
            if extracted_answer_str == "upregulated":
                return 1.0  # Score indicating differentially expressed
            elif extracted_answer_str == "downregulated":
                return 0.0
            elif extracted_answer_str == "not differentially expressed":
                return 2
            else: # None or other unexpected values from extract_answer
                return 0.5  # Neutral / Uncertain score

    # --- Initialize Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Process {accelerator.process_index} using device: {device}")

    # --- Load Model and Tokenizer (on main process first) ---
   # --- Load Model and Tokenizer (on main process first) ---
  # --- Initialize ChatOpenAI Client ---
    # Check for placeholder values specifically
        # exit(1)

    try:
        llm = ChatOpenAI(
            model=args.model_name,
            api_key=args.marketplace_api_key,
            base_url=args.marketplace_url,
            max_tokens=2048,
            max_retries=3,
            request_timeout=180,
        )

    except Exception as e:
        accelerator.print(f"Error initializing ChatOpenAI: {e}")
        exit(1)


    if args.AUROC_stage == "dir":
        print("args.AUROC_stage == dir")
        prompt_mode = "o3_direction_test"
    else:
        prompt_mode = "o3_test" 
    test_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, split="test", test_split_cell_lines = args.test_split_cell_lines, prompt_mode=prompt_mode)
    
    total_samples = len(test_dataset)
    sample_size = int(total_samples)
    indices = random.sample(range(total_samples), sample_size)
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Full test dataset size: {len(test_dataset)} samples.")

    def collate_fn(batch):
        prompts = [item.get('prompt', None) for item in batch]
        
        # MODIFICATION HERE:
 # For single_gene_prediction, or other tasks, assume solution_text is fine
        solutions = [item.get('solution', None) for item in batch] # Or whatever 'single_gene_prediction' expects

        perts = [item.get('pert', None) for item in batch]
        genes = [item.get('gene', None) for item in batch] # This 'gene' is likely for single_gene_prediction's target
        cell_types = [item.get('cell_type', None) for item in batch]

        valid_indices = [i for i, p in enumerate(prompts) if p is not None and isinstance(p, list) and len(p) > 0]
        if len(valid_indices) != len(prompts):
            warnings.warn(f"Batch contained {len(prompts) - len(valid_indices)} items with missing, invalid, or empty 'prompt' key.")
            prompts = [prompts[i] for i in valid_indices]
            solutions = [solutions[i] for i in valid_indices] # Filter solutions accordingly
            perts = [perts[i] for i in valid_indices]
            genes = [genes[i] for i in valid_indices]
            cell_types = [cell_types[i] for i in valid_indices]
        
        if not prompts:
            return None

        return {
            "prompts": prompts,
            "solutions": solutions, # Now 'solutions' will have the correct format per task
            "perts": perts,
            "genes": genes,
            "cell_types": cell_types,
        }

    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=32,
        pin_memory=False 
    )
    
    # Prepare model and dataloader
    test_dataloader = accelerator.prepare(test_dataloader)
    accelerator.print("DataLoader prepared.")
    # 3. Run Inference and Evaluate
    total_correct_local = 0
    total_evaluated_local = 0
    results_local = []
    samples_printed_count = 0

    accelerator.print(f"\nStarting distributed evaluation...")
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, file=sys.stdout)

    for batch_idx, batch in enumerate(progress_bar):
        if batch is None: # Skip batch if collate_fn returned None
            accelerator.print(f"Skipping empty or problematic batch {batch_idx} on Rank {accelerator.process_index}")
            continue

        # --- Get data from batch ---
        prompts = batch['prompts'] # List of chat message lists
        solutions = batch['solutions']
        perts = batch['perts']
        genes = batch['genes']
        cell_types = batch['cell_types']

        if not prompts: # Double check if batch somehow became empty
            accelerator.print(f"Skipping batch {batch_idx} due to empty prompts list after preparation on Rank {accelerator.process_index}")
            continue

        # --- Make API Call ---
        try:
            # Use llm.batch for parallel API calls within the batch
            # Pass config for batch-specific settings if needed (temperature already set in init)
            outputs = llm.batch(prompts) # Removed config if temperature is global
            # Extract content from AIMessage objects
            generated_texts = [msg.content if isinstance(msg, AIMessage) else str(msg) for msg in outputs]

        except Exception as e:
            accelerator.print(f"\nError during API call on Rank {accelerator.process_index}, Batch {batch_idx}: {e}")
            # Handle error: add placeholder results
            num_failed = len(prompts)
            # Don't increment total_evaluated_local here if we want accuracy excluding API errors
            for i in range(num_failed):
                user_prompt_content = "API_CALL_FAILED"
                # Safely access prompt content
                if i < len(prompts) and prompts[i] and isinstance(prompts[i], list) and len(prompts[i]) > 0:
                    user_msg = next((msg for msg in reversed(prompts[i]) if isinstance(msg, dict) and msg.get('role') == 'user'), None)
                    if user_msg: user_prompt_content = user_msg.get('content', 'Prompt content unavailable')
                    user_prompt_content = user_prompt_content[:200] + "... (API Call Failed)"

                results_local.append({
                    "user_prompt": user_prompt_content,
                    "pert": perts[i] if i < len(perts) else "N/A",
                    "gene": genes[i] if i < len(genes) else "N/A",
                    "cell_type": cell_types[i] if i < len(cell_types) else "N/A",
                    "generated_text": f"Error: {e}",
                    "extracted_answer": None,
                    "correct_solution": solutions[i] if i < len(solutions) else "N/A",
                    "is_correct_strict": False, # Mark as incorrect due to API failure
                    "api_error": True   # Add flag for error
                })
            # Update tqdm description with error info if needed
            if accelerator.is_local_main_process:
                current_postfix = progress_bar.postfix
                if isinstance(current_postfix, dict):
                    new_postfix = current_postfix.copy()
                else:
                    # If it's not a dict (e.g., None or a string), start fresh
                    # or try to preserve the description if it was in the postfix as a string key
                    new_postfix = {}
                    # If you suspect the description string itself might be what progress_bar.postfix is,
                    # you might not want to carry it over into the dict structure of set_postfix.
                    # For simplicity, let's assume postfix should only contain key-value pairs.

                new_postfix["api_errors"] = new_postfix.get("api_errors", 0) + num_failed
                progress_bar.set_postfix(new_postfix, refresh=True)
            time.sleep(2) # Add a small delay before retrying next batch after an error
            continue # Skip to the next batch

        # Compare generated answers with solutions locally
        for i, gen_text in enumerate(generated_texts):
            
            extracted_answer = extract_answer(gen_text)

            correct_solution = solutions[i]
            print(f"DEBUG: Correct solution: {correct_solution}")  # Debug print
            is_correct = (extracted_answer == correct_solution)
            print(is_correct)
            if is_correct:
                total_correct_local += 1
            total_evaluated_local += 1


            results_local.append({
                "pert": perts[i],               
                "gene": genes[i],               
                "cell_type": cell_types[i],     
                "generated_text": gen_text,
                "extracted_answer": extracted_answer,
                "correct_solution": correct_solution,
                "is_correct": is_correct
            })
        
        if hasattr(args, 'save_interval_batches') and args.save_interval_batches > 0 and \
           (batch_idx + 1) % args.save_interval_batches == 0:

            accelerator.print(f"Rank {accelerator.process_index}: Reached periodic save point at batch {batch_idx + 1}. Synchronizing...")
            accelerator.wait_for_everyone()

            # Gather current results_local from all processes.
            # Each item in gathered_intermediate_results will be the results_local list from one process.
            gathered_intermediate_results = gather_object(results_local)

            if accelerator.is_main_process:
                accelerator.print(f"Main process: Attempting periodic save for batch {batch_idx + 1}.")

                # Flatten the list of lists (or list of dicts) into a single list of dictionaries
                all_current_results_flat = []
                if gathered_intermediate_results and isinstance(gathered_intermediate_results, list):
                    if len(gathered_intermediate_results) > 0 and isinstance(gathered_intermediate_results[0], dict):
                        # Case: gather_object returned an already flat list (e.g., single process)
                        all_current_results_flat = gathered_intermediate_results
                    elif len(gathered_intermediate_results) > 0 and all(isinstance(sublist, list) for sublist in gathered_intermediate_results):
                        # Case: gather_object returned a list of lists (typical for multi-process)
                        for process_results_list in gathered_intermediate_results:
                            all_current_results_flat.extend(process_results_list)
                    # If gathered_intermediate_results is empty or has an unexpected structure,
                    # all_current_results_flat might remain empty. This is handled by the next check.

                if all_current_results_flat:
                    # Ensure output directory exists
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # output_path was defined at the start of main

                    accelerator.print(f"Main process: Periodically saving {len(all_current_results_flat)} accumulated results to {output_path}...")
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f: # Overwrite with the current snapshot
                            for result_item in all_current_results_flat:
                                if isinstance(result_item, dict): # Double-check item is a dict
                                    f.write(json.dumps(result_item) + '\n')
                        accelerator.print(f"Main process: Periodic save to {output_path} successful.")
                    except Exception as e:
                        accelerator.print(f"Main process: Error during periodic save to {output_path}: {e}")
                else:
                    accelerator.print(f"Main process: No results to save periodically at batch {batch_idx + 1} (results list was empty or not in expected format).")

            # Synchronize after the main process attempts saving
            accelerator.print(f"Rank {accelerator.process_index}: Synchronizing after periodic save attempt at batch {batch_idx + 1}.")
            accelerator.wait_for_everyone()
        # --- END MODIFICATION: Periodic Saving ---
    # 4. Gather Results Across All Processes
    # After the evaluation loop
    progress_bar.close() # Ensure tqdm is closed
    accelerator.print(f"Rank {accelerator.process_index} finished evaluation loop. Local correct: {total_correct_local}, Local evaluated: {total_evaluated_local}. Waiting at barrier BEFORE reduce...")
    accelerator.wait_for_everyone()
    accelerator.print(f"Rank {accelerator.process_index} passed barrier AFTER loop, BEFORE reduce.")

    correct_tensor = torch.tensor(total_correct_local, device=accelerator.device)
    evaluated_tensor = torch.tensor(total_evaluated_local, device=accelerator.device)

    accelerator.print(f"Rank {accelerator.process_index} attempting first reduce (correct_tensor).")
    total_correct_gathered = accelerator.reduce(correct_tensor, reduction="sum")
    accelerator.print(f"Rank {accelerator.process_index} completed first reduce. Waiting at barrier BEFORE second reduce...")
    accelerator.wait_for_everyone() # Add another barrier
    accelerator.print(f"Rank {accelerator.process_index} passed barrier AFTER first reduce, BEFORE second reduce.")

    accelerator.print(f"Rank {accelerator.process_index} attempting second reduce (evaluated_tensor).")
    total_evaluated_gathered = accelerator.reduce(evaluated_tensor, reduction="sum")
    accelerator.print(f"Rank {accelerator.process_index} completed second reduce.")

    # Gather the detailed results list (using correct method)
    accelerator.print(f"Rank {accelerator.process_index}: Gathering detailed results objects (results_local size: {len(results_local)}).")
    gathered_results_list = gather_object(results_local)
    accelerator.print(f"Rank {accelerator.process_index}: Completed gather_object. Waiting at BARRIER 4.")
    accelerator.wait_for_everyone() # BARRIER 4
    accelerator.print(f"Rank {accelerator.process_index}: Passed BARRIER 4.")

    # --- Process and Save Results (only on main process) ---
    if accelerator.is_main_process:
            total_correct = total_correct_gathered.item()
            total_evaluated = total_evaluated_gathered.item()

            if total_evaluated > 0:
                overall_accuracy_percentage = (total_correct / total_evaluated) * 100
                print("\n--- Evaluation Complete ---")
                print(f"Total Samples Evaluated (Global): {total_evaluated}")
                print(f"Correct Predictions (Global): {total_correct}")
                print(f"Baseline Accuracy (Global): {overall_accuracy_percentage:.2f}%")
            else:
                print("\nNo samples were evaluated globally.")
                overall_accuracy_percentage = 0

            all_results = []
            # (Your existing logic to populate all_results based on gathered_results_list - crucial for this to work)
            # Ensure all_results is a flat list of dictionaries here.
            # Example:
            if gathered_results_list and isinstance(gathered_results_list, list):
                if gathered_results_list and len(gathered_results_list) > 0 and isinstance(gathered_results_list[0], dict):
                    all_results = gathered_results_list
                elif gathered_results_list and len(gathered_results_list) > 0 and all(isinstance(sublist, list) for sublist in gathered_results_list):
                    for process_results in gathered_results_list:
                        all_results.extend(process_results)
                elif not gathered_results_list:
                    all_results = []
                else:
                    print(f"Warning: gathered_results_list structure is unexpected. Results processing might be incomplete.")
                    all_results = [] # Or handle as per your debugging
            else:
                print(f"No detailed results were gathered. Skipping JSONL save and detailed stats.")
                all_results = []


            if all_results:
                # (Your JSONL saving logic - assumed to be here and working)
                print(f"Processed {len(all_results)} detailed results for saving.")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / output_filename
                print(f"Saving detailed JSONL results to {output_path}...")
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for result_item in all_results:
                            if isinstance(result_item, dict):
                                f.write(json.dumps(result_item) + '\n')
                    print("JSONL Results saved successfully.")
                except Exception as e:
                    print(f"Error saving JSONL results to file: {e}")


            # --- Calculate and Save Detailed Statistics to a .txt file ---
            if all_results and total_evaluated > 0:
                stats_string_parts = []

                # 1. Overall Summary
                stats_string_parts.append("--- Overall Evaluation Summary ---")
                stats_string_parts.append(f"Total Samples Evaluated: {total_evaluated}")
                stats_string_parts.append(f"Correct Predictions: {total_correct}")
                stats_string_parts.append(f"Overall Accuracy: {overall_accuracy_percentage:.2f}%\n")

                # Prepare lists for classification_report
                y_true = []
                y_pred_for_report = []
                
                # Define the classes. "none_extracted" is for cases where your extract_answer returned None.
                # These labels must match what's in correct_solution and the (mapped) extracted_answer.
                defined_classes = ["upregulated", "downregulated", "not differentially expressed"]
                report_labels = defined_classes + ["none_extracted"] # For predictions

                cell_type_stats = {}
                pert_stats = {}
                prediction_distribution = {
                    "upregulated": 0,
                    "downregulated": 0,
                    "not differentially expressed": 0,
                    "None/Other": 0 # For raw extracted answers that were None
                }

                for item in all_results:
                    true_label = item.get('correct_solution')
                    print("correct_solution:", true_label)
                    extracted_ans = item.get('extracted_answer')

                    y_true.append(true_label)
                    
                    # Map extracted_answer for the report (especially None values)
                    if extracted_ans in defined_classes:
                        y_pred_for_report.append(extracted_ans)
                    else: # This includes None or any other unexpected string
                        y_pred_for_report.append("none_extracted")

                    # For prediction_distribution (raw counts including None before mapping for report)
                    if extracted_ans in prediction_distribution:
                        prediction_distribution[extracted_ans] += 1
                    else: # Catches None from extracted_answer, or other unexpected values
                        prediction_distribution["None/Other"] += 1
                    
                    # For other stats (cell_type, pert)
                    cell_type = item.get('cell_type', 'Unknown_Cell_Type')
                    pert = item.get('pert', 'Unknown_Perturbation')
                    is_correct = item.get('is_correct', False)

                    print("is_correct:", is_correct)
                    if cell_type not in cell_type_stats: cell_type_stats[cell_type] = {'correct': 0, 'total': 0}
                    cell_type_stats[cell_type]['total'] += 1
                    if is_correct: cell_type_stats[cell_type]['correct'] += 1

                    if pert not in pert_stats: pert_stats[pert] = {'correct': 0, 'total': 0}
                    pert_stats[pert]['total'] += 1
                    if is_correct: pert_stats[pert]['correct'] += 1

                # 2. Classification Report
                stats_string_parts.append("--- Classification Report ---")

                # --- Generate and Save Classification Report SEPARATELY ---
                classification_report_filename = "classification_report.txt"
                classification_report_path = output_dir / classification_report_filename
                # Ensure y_true and y_pred_for_report are not empty
                if y_true and y_pred_for_report:
                    # 'labels' ensures all desired classes are in the report, even if some have 0 support in predictions.
                    # 'target_names' provides the names for these labels.
                    # 'zero_division=0' prevents warnings if a metric is undefined (e.g., precision for a class never predicted).
                    class_report_str = classification_report(
                        y_true,
                        y_pred_for_report,
                        labels=report_labels, # Use all potential prediction labels
                        target_names=report_labels, # Names for these labels
                        zero_division=0,
                        digits=2 # Number of digits for floating point numbers
                    )
                else:
                    class_report_str.append("  Not enough data to generate classification report.")
                # Save classification report to file

                with open(classification_report_path, 'w', encoding='utf-8') as f:
                    f.write(class_report_str)
                print(f"Classification report saved to {classification_report_path}")


                # 3. Predicted Answer Distribution (Raw, as extracted)
                stats_string_parts.append("--- Predicted Answer Distribution (Model Output) ---")
                if total_evaluated > 0:
                    for answer_type, count in sorted(prediction_distribution.items()):
                        percentage = (count / total_evaluated) * 100
                        stats_string_parts.append(f"  Predicted as '{answer_type}': {count} times ({percentage:.2f}%)")
                else:
                    stats_string_parts.append("  No samples evaluated for distribution.")
                stats_string_parts.append("\n")

                # 4. Accuracy by Cell Type
                stats_string_parts.append("--- Accuracy by Cell Type ---")
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
                try:
                    with open(stats_output_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(stats_string_parts))
                    print(f"Detailed statistics saved to {stats_output_path}")
                except Exception as e:
                    print(f"Error saving statistics to file: {e}")

            elif total_evaluated == 0:
                print("Skipping detailed statistics calculation as no samples were evaluated globally.")
            elif not all_results:
                print("Skipping detailed statistics calculation as no valid results were processed into all_results.")

        # Ensure all processes finish
    accelerator.wait_for_everyone()
