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
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset

# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

# --- Configuration ---
def main(args):
    print("Starting tuples evaluation script...")


    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
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
                return None
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

    if accelerator.is_main_process:
        print("Model and Tokenizer loaded on main process.")

    # 2. Load Test Dataset
    if accelerator.is_main_process:
        print("Loading test dataset definition...")
    if args.generate_all_non_de_samples or args.generate_4x_non_de_samples:
        prompt_mode = "warning_different_distributions"
    elif args.gene_enrichment:
        prompt_mode = "gene_enrichment"
    
    else:
        prompt_mode = "default"
    test_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, prompt_mode=prompt_mode, test_split_cell_lines= args.test_split_cell_lines, split="test", generate_all_non_de_samples=args.generate_all_non_de_samples, generate_4x_non_de_samples=args.generate_4x_non_de_samples)

    total_samples = len(test_dataset)
    sample_size = int(total_samples * args.dataset_fraction)  # Use the dataset_fraction argument
    indices = random.sample(range(total_samples), sample_size)
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    if accelerator.is_main_process:
         print(f"Full test dataset size: {len(test_dataset)} samples.")

    # --- Collate function (Corrected) ---
    def collate_fn(batch):
        # Extract necessary fields, use .get for safety with metadata
        prompts = [item.get('prompt', None) for item in batch] # Expects 'prompt' key now
        solutions = [item.get('solution', None) for item in batch]
        perts = [item.get('pert', None) for item in batch]
        genes = [item.get('gene', None) for item in batch]
        cell_types = [item.get('cell_type', None) for item in batch]

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
            "original_prompts": prompts, # Keep original prompts for context if needed
            "perts": perts,             # <<< Pass metadata through
            "genes": genes,             # <<< Pass metadata through
            "cell_types": cell_types,   # <<< Pass metadata through
        }

    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Prepare model and dataloader
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # 3. Run Inference and Evaluate
    total_correct_local = 0
    total_evaluated_local = 0
    results_local = []
    samples_printed_count = 0

    accelerator.print(f"\nStarting distributed evaluation...")
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, file=sys.stdout)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
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
                # Optionally skip to next batch or handle error
                continue # Skip batch on generation error


            generated_ids = outputs[:, input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # print("generated_texts:", generated_texts)  # Debugging output
            # Compare generated answers with solutions locally
            for i, gen_text in enumerate(generated_texts):
                extracted_answer = extract_answer(gen_text)
                correct_solution = solutions[i]
                is_correct = (extracted_answer == correct_solution)

                if is_correct:
                    total_correct_local += 1
                total_evaluated_local += 1

                # Append results including metadata
                user_prompt_content = ""
                if original_prompts[i] and isinstance(original_prompts[i], list) and len(original_prompts[i]) > 1:
                   user_prompt_content = original_prompts[i][-1].get('content', "")

                # print("-----")
                # print({
                #     "user_prompt": user_prompt_content,
                #     "pert": perts[i],               # <<< Now available
                #     "gene": genes[i],               # <<< Now available
                #     "cell_type": cell_types[i],     # <<< Now available
                #     "generated_text": gen_text,
                #     "extracted_answer": extracted_answer,
                #     "correct_solution": correct_solution,
                #     "is_correct": is_correct
                # })
                # print("-------")
                results_local.append({
                    "user_prompt": user_prompt_content,
                    "pert": perts[i],               # <<< Now available
                    "gene": genes[i],               # <<< Now available
                    "cell_type": cell_types[i],     # <<< Now available
                    "generated_text": gen_text,
                    "extracted_answer": extracted_answer,
                    "correct_solution": correct_solution,
                    "is_correct": is_correct
                })

    # 4. Gather Results Across All Processes
    progress_bar.close() 

    accelerator.print(f"Rank {accelerator.process_index} finished evaluation. Gathering results...")
    
    # This single call will collect the `results_local` list from every process
    # and create a list of lists on the main process. It's synchronized.
    gathered_results = gather_object(results_local)

    # The rest of the script now only runs on the main process.
    if accelerator.is_main_process:
        print("\n--- Aggregating results from all processes ---")
        
        # Flatten the list of lists into a single list of results
        all_results = [item for sublist in gathered_results for item in sublist]

        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
    accelerator.print(f"Rank {accelerator.process_index} finished and saved temporary results.")
    
    # Wait for all processes to finish writing their files.
    accelerator.wait_for_everyone()

    # --- Process and Save Results (only on main process) ---
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

        # --- Calculate and Save Detailed Statistics ---
        if all_results:
            stats_string_parts = []
            
            # Calculate overall metrics from the combined list
            total_evaluated = len(all_results)
            total_correct = sum(1 for item in all_results if item['is_correct'])
            overall_accuracy_percentage = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0

            # 1. Overall Summary
            stats_string_parts.append("--- Overall Evaluation Summary ---")
            stats_string_parts.append(f"Total Samples Evaluated: {total_evaluated}")
            stats_string_parts.append(f"Correct Predictions: {total_correct}")
            stats_string_parts.append(f"Overall Accuracy: {overall_accuracy_percentage:.2f}%\n")
            
            y_true = [item.get('correct_solution') for item in all_results]
            y_pred_raw = [item.get('extracted_answer') for item in all_results]
            
            defined_classes = ["upregulated", "downregulated", "not differentially expressed"]
            report_labels = defined_classes + ["none_extracted"]

            y_pred_for_report = [pred if pred in defined_classes else "none_extracted" for pred in y_pred_raw]

            # 2. Classification Report
            classification_report_filename = "classification_report.txt"
            classification_report_path = output_dir / classification_report_filename
            class_report_str = "" # Initialize as string
            if y_true and y_pred_for_report:
                class_report_str = classification_report(
                    y_true, y_pred_for_report,
                    labels=report_labels, target_names=report_labels,
                    zero_division=0, digits=2
                )
            else:
                ### CHANGE 3: Assign string directly, don't append to a list
                class_report_str = "  Not enough data to generate classification report."

            with open(classification_report_path, 'w', encoding='utf-8') as f:
                f.write(class_report_str)
            print(f"Classification report saved to {classification_report_path}")

            # (The rest of your statistics generation code was well-structured and can remain)
            # ... [Your code for prediction distribution, accuracy by cell type, etc.] ...
            # 3. Predicted Answer Distribution (Raw, as extracted)
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

