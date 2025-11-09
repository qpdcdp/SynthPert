from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import random
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import numpy as np
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

# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset

# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

# --- Configuration ---
def main(args):

    if args.checkpoint_path:
        # Load the model from the specified checkpoint
        model_name_or_path = args.checkpoint_path
    else:
        if "deepseek" in args.model_name.lower():

            model_name_or_path = "deepseek-ai/" + args.model_name
        elif "qwen" in args.model_name.lower():
            model_name_or_path = "Qwen/" + args.model_name
        elif "gemma" in args.model_name.lower():
            model_name_or_path = "google/" + args.model_name
        else:
            model_name_or_path = args.model_name

    print("checkpoint_path")
    csv_data_directory = args.csv_data_directory
    output_dir = Path(args.output_dir)
    output_filename = "responses" \
    ".jsonl" # Define filename separately

    batch_size = args.batch_size # batch size per device
    batches_to_print = 0 # How many batches to print for debugging

    # --- Helper Function to Parse Answer ---
    # def extract_answer(generated_text):
    #     match = re.search(r"</think>(.*)", generated_text, re.IGNORECASE | re.DOTALL)
    #     if match:
    #         answer = match.group(1).strip().lower()
    #         print(f"DEBUG: Extracted answer: {answer}")  # Debug print
    #         if answer in ["upregulated", "downregulated", "not differentially expressed"]: return answer
    #         else:
    #             if "upregulated" in answer: return "upregulated"
    #             if "downregulated" in answer: return "downregulated"
    #             if "not differentially expressed" in answer: return "not differentially expressed"
    #             return None
    #     return None

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

    # --- NEW Helper Functions for AUROC ---
    if args.AUROC_stage == "dir":
        print (f"AUROC stage is 'dir'. Using direction_test_prompt.")

        def map_solution_to_binary_label(solution_str):
            if solution_str == "upregulated":
                return 1  # Differentially Expressed
            elif solution_str == "downregulated":
                return 0  # Not Differentially Expressed
            return -1 # Should not happen if data is clean

        def map_prediction_to_score(extracted_answer_str):
            if extracted_answer_str == "upregulated":
                return 1.0  # Score indicating differentially expressed
            elif extracted_answer_str == "downregulated":
                return 0.0  # Score indicating not differentially expressed
            else: # None or other unexpected values from extract_answer
                return 0.5  # Neutral / Uncertain score
    else:
        def map_solution_to_binary_label(solution_str):
            if solution_str in ["upregulated", "downregulated"]:
                return 1  # Differentially Expressed
            elif solution_str == "not differentially expressed":
                return 0  # Not Differentially Expressed
            return -1 # Should not happen if data is clean

        def map_prediction_to_score(extracted_answer_str):
            if extracted_answer_str in ["upregulated", "downregulated"]:
                return 1.0  # Score indicating differentially expressed
            elif extracted_answer_str == "not differentially expressed":
                return 0.0  # Score indicating not differentially expressed
            else: # None or other unexpected values from extract_answer
                return 0.5  # Neutral / Uncertain score


    # --- Initialize Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Process {accelerator.process_index} using device: {device}")

    # --- Load Model and Tokenizer (on main process first) ---
   # --- Load Model and Tokenizer (on main process first) ---
    # The `model` variable will be defined within this block
    model = None # Initialize to ensure it's defined

    with accelerator.main_process_first():
        accelerator.print(f"DEBUG: ---- Starting Model and Tokenizer Loading ----")
        accelerator.print(f"DEBUG: model_name_or_path = {model_name_or_path}")
        accelerator.print(f"DEBUG: args.lora_checkpoint = {args.lora_checkpoint}")

        # Tokenizer loading (uses model_name_or_path if args.lora_checkpoint is None,
        # or if args.lora_checkpoint is set, model_name_or_path should be the base model name)
        # For clarity, let's define tokenizer_path explicitly
        tokenizer_path = model_name_or_path
        if args.lora_checkpoint and args.checkpoint_path:
             # If both are set, it's a bit ambiguous.
             # Assume args.checkpoint_path is the *base model* when lora_checkpoint is also given.
             # Or, if args.checkpoint_path is the LoRA adapter itself, then model_name_or_path
             # (derived from args.model_name) should be the base.
             # Let's stick to the logic: if lora_checkpoint is present, model_name_or_path
             # is treated as the base model.
             accelerator.print(f"DEBUG: Loading tokenizer from base model path: {model_name_or_path} (when LoRA is used)")
             tokenizer_path = model_name_or_path
        elif args.checkpoint_path and not args.lora_checkpoint:
             accelerator.print(f"DEBUG: Loading tokenizer from checkpoint path: {args.checkpoint_path}")
             tokenizer_path = args.checkpoint_path # This is final_merged_model, tokenizer should be there.

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            if accelerator.is_main_process:
                print("Warning: Tokenizer missing pad token; setting to eos_token.")
            tokenizer.pad_token = xxxx
        tokenizer.padding_side = "left"

        if args.lora_checkpoint:
            # Case 1: Loading a base model and applying a LoRA adapter
            accelerator.print(f"DEBUG: Loading base model from: {args.lora_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                args.lora_checkpoint, # This should be the BASE model identifier
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
            )
            # if accelerator.is_main_process:
            #     accelerator.print(f"DEBUG: Base model loaded. config.use_cache = {base_model_for_lora.config.use_cache}")

            # from peft import PeftModel
            # accelerator.print(f"DEBUG: Applying LoRA adapter from: {args.lora_checkpoint}")
            # model_with_adapter = PeftModel.from_pretrained(base_model_for_lora, args.lora_checkpoint)

            # accelerator.print(f"DEBUG: Merging LoRA adapter...")
            # model = model_with_adapter.merge_and_unload()
            # if accelerator.is_main_process:
            #     accelerator.print(f"DEBUG: Model after LoRA merge. config.use_cache = {model.config.use_cache}")

        else:
            # Case 2: Loading a pre-merged model (e.g., final_merged_model)
            # Here, model_name_or_path is args.checkpoint_path
            accelerator.print(f"DEBUG: Loading pre-merged model from: {model_name_or_path}")

            # --- THIS IS THE KEY PART FOR THE "SLOWER" SCENARIO ---
            # Load the config first to inspect and modify
            original_config = AutoConfig.from_pretrained(model_name_or_path)
            if accelerator.is_main_process:
                accelerator.print(f"DEBUG: Original config.use_cache from {model_name_or_path}: {original_config.use_cache}")



            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=original_config, # Pass the (potentially modified) config
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
            )
            if accelerator.is_main_process:
                accelerator.print(f"DEBUG: Pre-merged model loaded. config.use_cache = {model.config.use_cache}")

        if model is not None: # Ensure model was actually loaded
            model.eval()
        else:
            raise ValueError("Model was not loaded properly!")

        accelerator.print(f"DEBUG: ---- Model and Tokenizer Loading Complete ----")


    if accelerator.is_main_process:
        print("Model and Tokenizer loaded on main process.")
        if model: # Check again after the with block
            print(f"FINAL CHECK (main process): model.config.use_cache = {model.config.use_cache}")


    if accelerator.is_main_process:
        print("Model and Tokenizer loaded on main process.")

    # 2. Load Test Dataset
    if accelerator.is_main_process:
        print("Loading test dataset definition...")
    
    if args.AUROC_stage == "dir":
        print("args.AUROC_stage == dir")
        prompt_mode = "direction_test_prompt"
    else:
        prompt_mode = "default"    
    test_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, split="test", test_split_cell_lines = args.test_split_cell_lines, prompt_mode=prompt_mode)
        # Add these lines to sample 5% of the data:
    total_samples = len(test_dataset)
    sample_size = int(total_samples * 0.25)  # 50% of the data
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
            print("prompts", prompts)
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
        collate_fn=collate_fn, num_workers=4 # You can adjust num_workers
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

            # # Debug Print Block (Main Process Only)
            # if accelerator.is_main_process and batch_idx < batches_to_print:
            #     print(f"\n--- Batch {batch_idx} Generations (Rank 0) ---")
            #     for i in range(len(generated_texts)):
            #         extracted_answer = extract_answer(generated_texts[i]) # Extract here too for printing
            #         user_prompt_content = "" # Define before use
            #         if original_prompts[i] and isinstance(original_prompts[i], list) and len(original_prompts[i]) > 1:
            #             user_prompt_content = original_prompts[i][-1].get('content', "")

            #         print(f"Sample {samples_printed_count}:")
            #         print(f"  User Prompt: {user_prompt_content[:200]}...")
            #         print(f"  Expected: {solutions[i]}")
            #         print(f"  Generated Text: >>>{generated_texts[i]}<<<")
            #         print(f"  Extracted Answer: >>>{extracted_answer}<<<")
            #         print("-" * 20)
            #         samples_printed_count += 1 # Increment only when printing


            # Compare generated answers with solutions locally
            for i, gen_text in enumerate(generated_texts):
                # print("gen_text", repr(gen_text))
                
                extracted_answer = extract_answer(gen_text)
                # print("extracted_answer")
                # print(extracted_answer)
                correct_solution = solutions[i]
                is_correct = (extracted_answer == correct_solution)

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


    # ==============================================================================
    # --- AUROC CALCULATION ON RANDOM SUBSETS (NEW SECTION) ---
    # ==============================================================================

    auroc_string_parts = []
    num_auroc_subsets = 3
    auroc_subset_percentage = 0.20 # Using 50% as per your example in the question text. Change to 0.05 for 5%

    auroc_all_runs_overall = []
    auroc_all_runs_per_cell_type = [] # List of dictionaries for each run

    # Load the full base test dataset once for subsetting
    # Ensure csv_data_directory is defined
    if accelerator.is_main_process:
        print(f"\n--- Starting AUROC Evaluation on {num_auroc_subsets} Random Subsets ({auroc_subset_percentage*100:.0f}% each) ---")
    if args.AUROC_stage == "dir":
        print("args.AUROC_stage == dir")
        prompt_mode = "direction_test_prompt"
    else:
        prompt_mode = "default"    
    # Base dataset for creating subsets
    try:
        with accelerator.main_process_first(): # Ensure dataset is loaded once
            full_test_dataset_for_auroc = DiffExpressionDataset(csv_dir=csv_data_directory, split="test", prompt_mode=prompt_mode, test_split_cell_lines = args.test_split_cell_lines)
    except Exception as e:
        accelerator.print(f"Error loading full_test_dataset_for_auroc: {e}. Skipping AUROC calculation.")
        full_test_dataset_for_auroc = None

    if full_test_dataset_for_auroc:
        total_samples_full = len(full_test_dataset_for_auroc)
        sample_size = int(total_samples_full * auroc_subset_percentage)

        for i_subset in range(num_auroc_subsets):
            accelerator.print(f"\n--- AUROC Subset {i_subset+1}/{num_auroc_subsets} ---")

            # 1. Create a random subset of the dataset
            # Ensure this sampling happens on the main process first or is consistent across processes
            # For simplicity and consistency, let's generate indices on main and broadcast or ensure same seed
            # However, DataLoader with DistributedSampler will handle sharding the subset correctly.
            current_subset_indices = random.sample(range(total_samples_full), sample_size)
            current_test_subset = torch.utils.data.Subset(full_test_dataset_for_auroc, current_subset_indices)
            
            accelerator.print(f"Subset {i_subset+1} size: {len(current_test_subset)} samples.")

            # 2. Create DataLoader for this subset
            # batch_size is defined earlier in your script
            auroc_subset_dataloader = DataLoader(
                current_test_subset, batch_size=batch_size, shuffle=False, # No shuffle for eval
                collate_fn=collate_fn, num_workers=4 # Adjust num_workers as needed
            )

            # 3. Prepare DataLoader
            auroc_subset_dataloader_prepared = accelerator.prepare(auroc_subset_dataloader)
            
            # 4. Inference loop for this subset
            subset_results_local = [] # Store dicts: {'true_label': 0/1, 'pred_score': float, 'cell_type': str}

            # Ensure model is in eval mode (should be already, but good practice)
            # unwrapped_model = accelerator.unwrap_model(model) # model is already prepared
            # unwrapped_model.eval() # Should already be in eval mode

            progress_bar_auroc = tqdm(
                auroc_subset_dataloader_prepared,
                desc=f"Rank {accelerator.process_index} AUROC Subset {i_subset+1}",
                disable=not accelerator.is_local_main_process,
                file=sys.stdout
            )

            with torch.no_grad():
                for batch_idx, batch in enumerate(progress_bar_auroc):
                    if batch is None:
                        accelerator.print(f"Skipping empty/problematic batch {batch_idx} in AUROC subset {i_subset+1}")
                        continue

                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    # These are original string solutions from the dataset via collate_fn
                    original_solutions_str = batch['solutions']
                    cell_types_str = batch['cell_types'] # From collate_fn

                    unwrapped_model_ref = accelerator.unwrap_model(model) # Get the underlying model
                    try:
                        outputs = unwrapped_model_ref.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=2048, # Or a smaller value if appropriate
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=True, # Or False if you prefer greedy for AUROC scores
                            temperature=1,  # Consider temperature if do_sample=True
                        )
                    except Exception as e:
                        accelerator.print(f"Error during model.generate in AUROC loop (Rank {accelerator.process_index}, Batch {batch_idx}): {e}")
                        continue
                    
                    generated_ids = outputs[:, input_ids.shape[1]:]
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    for i, gen_text in enumerate(generated_texts):
                        extracted_pred_str = extract_answer(gen_text)
                        true_solution_str = original_solutions_str[i]
                        
                        binary_true_label = map_solution_to_binary_label(true_solution_str)
                        pred_score = map_prediction_to_score(extracted_pred_str)
                        
                        if binary_true_label != -1: # Only include if valid true label
                            subset_results_local.append({
                                'true_label_binary': binary_true_label,
                                'predicted_score': pred_score,
                                'cell_type': cell_types_str[i]
                            })
            
            # Gather results from all processes for this subset
            gathered_subset_results_list = gather_object(subset_results_local)

            # 5. Calculate AUROC on the main process for this subset
            if accelerator.is_main_process:
                flat_subset_results = []
                if gathered_subset_results_list and isinstance(gathered_subset_results_list, list):
                    for res_list in gathered_subset_results_list:
                        if isinstance(res_list, list):
                            flat_subset_results.extend(res_list)
                        elif isinstance(res_list, dict): # If gather_object returns flat list already
                            flat_subset_results.append(res_list)
                
                # If gather_object might return a flat list directly when num_processes=1
                if not flat_subset_results and isinstance(gathered_subset_results_list, list) and \
                   len(gathered_subset_results_list) > 0 and isinstance(gathered_subset_results_list[0], dict):
                    flat_subset_results = gathered_subset_results_list


                if not flat_subset_results:
                    print(f"No results gathered for AUROC calculation in subset {i_subset+1}.")
                    auroc_all_runs_overall.append(None) # Placeholder
                    auroc_all_runs_per_cell_type.append({}) # Placeholder
                    continue

                y_true_subset = np.array([item['true_label_binary'] for item in flat_subset_results])
                y_scores_subset = np.array([item['predicted_score'] for item in flat_subset_results])
                cell_types_subset = [item['cell_type'] for item in flat_subset_results]

                current_run_auroc_overall = None
                current_run_auroc_per_cell_type = {}

                # Overall AUROC for this subset
                try:
                    if len(np.unique(y_true_subset)) > 1 and len(y_true_subset) > 1: # Need at least 2 samples and both classes
                        current_run_auroc_overall = roc_auc_score(y_true_subset, y_scores_subset)
                        auroc_all_runs_overall.append(current_run_auroc_overall)
                        print(f"  Subset {i_subset+1} Overall AUROC: {current_run_auroc_overall:.4f}")
                    else:
                        print(f"  Subset {i_subset+1} Overall AUROC: Not calculable (single class or <2 samples). y_true unique: {np.unique(y_true_subset)}")
                        auroc_all_runs_overall.append(None)
                except ValueError as e:
                    print(f"  Subset {i_subset+1} Overall AUROC: Error - {e}")
                    auroc_all_runs_overall.append(None)

                # Per-cell-type AUROC for this subset
                unique_cell_types = sorted(list(set(cell_types_subset)))
                for ct in unique_cell_types:
                    ct_indices = [j for j, cell in enumerate(cell_types_subset) if cell == ct]
                    y_true_ct = y_true_subset[ct_indices]
                    y_scores_ct = y_scores_subset[ct_indices]

                    try:
                        if len(np.unique(y_true_ct)) > 1 and len(y_true_ct) > 1:
                            auroc_ct = roc_auc_score(y_true_ct, y_scores_ct)
                            current_run_auroc_per_cell_type[ct] = auroc_ct
                            print(f"    Cell Type '{ct}' AUROC: {auroc_ct:.4f} (samples: {len(y_true_ct)})")
                        else:
                            current_run_auroc_per_cell_type[ct] = None
                            print(f"    Cell Type '{ct}' AUROC: Not calculable (single class or <2 samples). Samples: {len(y_true_ct)}, y_true unique: {np.unique(y_true_ct)}")
                    except ValueError as e:
                        current_run_auroc_per_cell_type[ct] = None
                        print(f"    Cell Type '{ct}' AUROC: Error - {e}")
                auroc_all_runs_per_cell_type.append(current_run_auroc_per_cell_type)
            
            accelerator.wait_for_everyone() # Ensure all processes finish before next subset

    # --- Append AUROC results to the statistics string parts (on main process) ---
    if accelerator.is_main_process:
        # Assume auroc_string_parts is a list of strings from your previous stats saving
        # If not, initialize it: auroc_string_parts = []
        # This should ideally be the same list you used for accuracy stats.

        auroc_string_parts.append("\n--- AUROC Evaluation (Differentially Expressed vs. Not) ---")
        auroc_string_parts.append(f"Number of random subsets: {num_auroc_subsets}")
        auroc_string_parts.append(f"Subset size: {auroc_subset_percentage*100:.0f}% of total test data")

        # Overall AUROC summary
        valid_overall_aurocs = [val for val in auroc_all_runs_overall if val is not None]
        if valid_overall_aurocs:
            mean_overall_auroc = np.mean(valid_overall_aurocs)
            std_overall_auroc = np.std(valid_overall_aurocs)
            auroc_string_parts.append(f"\nMean Overall AUROC across subsets: {mean_overall_auroc:.4f} (Std: {std_overall_auroc:.4f})")
            auroc_string_parts.append(f"Individual Overall AUROCs: {[f'{val:.4f}' if val is not None else 'N/A' for val in auroc_all_runs_overall]}")
        else:
            auroc_string_parts.append("\nNo valid Overall AUROC scores calculated across subsets.")

        # Per-cell-type AUROC summary (Averaged)
        auroc_string_parts.append("\nMean Per-Cell-Type AUROC across subsets:")
        # Consolidate cell types seen across all runs
        all_seen_cell_types_for_auroc = set()
        for run_data in auroc_all_runs_per_cell_type:
            all_seen_cell_types_for_auroc.update(run_data.keys())
        
        if not all_seen_cell_types_for_auroc:
            auroc_string_parts.append("  No per-cell-type AUROC data available.")
        else:
            for ct in sorted(list(all_seen_cell_types_for_auroc)):
                ct_aurocs_across_runs = []
                for run_data in auroc_all_runs_per_cell_type:
                    if ct in run_data and run_data[ct] is not None:
                        ct_aurocs_across_runs.append(run_data[ct])
                
                if ct_aurocs_across_runs:
                    mean_ct_auroc = np.mean(ct_aurocs_across_runs)
                    std_ct_auroc = np.std(ct_aurocs_across_runs)
                    auroc_string_parts.append(f"  {ct}: {mean_ct_auroc:.4f} (Std: {std_ct_auroc:.4f}, Runs: {len(ct_aurocs_across_runs)}/{num_auroc_subsets})")
                else:
                    auroc_string_parts.append(f"  {ct}: Not calculable in any valid subset run.")
        
        auroc_string_parts.append("\n--- End of AUROC Evaluation ---")
        
        # Save AUROC statistics to a file
        roc_stats_filename = "roc_statistics.txt"
        roc_stats_path = output_dir / roc_stats_filename 

        with open(roc_stats_path, 'w', encoding='utf-8') as f: # 'w' to overwrite, or 'a' to append if separate
            f.write("\n".join(auroc_string_parts))
        print(f"Detailed statistics (including AUROC) saved to {roc_stats_path}")
