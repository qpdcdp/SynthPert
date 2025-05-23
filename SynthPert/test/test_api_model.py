# --- Imports ---
import os # For environment variables
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

# Assuming DiffExpressionDataset is correctly importable
from src.data import DiffExpressionDataset, GeneRegulationListDataset

# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO)
# Reduce httpx verbosity unless debugging API calls specifically
logging.getLogger('httpx').setLevel(logging.WARNING) # Changed from DEBUG to WARNING

def calculate_set_performance_metrics(pred_set, true_set):
    """
    Calculates TP, FP, FN, Precision, Recall, F1-score, and counts 
    for a pair of predicted and true gene sets.
    Assumes pred_set and true_set are actual sets of strings.
    """
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    num_pred = len(pred_set)
    num_true = len(true_set)

    # Handle cases for precision, recall, f1
    # If both predicted and true are empty, it's a perfect match.
    if num_pred == 0 and num_true == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / num_pred if num_pred > 0 else 0.0
        recall = tp / num_true if num_true > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "num_pred": num_pred, # Count of items in the predicted set
        "num_true": num_true,   # Count of items in the true set
        "precision": precision, # Fraction of predicted items that are correct
        "recall": recall,       # Fraction of true items that were predicted
        "f1": f1                # Harmonic mean of precision and recall
    }

def main(args):
    # --- Configuration ---
    # --- OpenAI API Configuration ---
    # *** IMPORTANT: Set these environment variables or replace the placeholders ***
    API_KEY = os.getenv("MARKETPLACE_API_KEY", "sk-7lN2yHKbX5NWkbzjFU0faQ") # Use getenv for flexibility
    BASE_URL = os.getenv("MARKETPLACE_BASE_URL", "https://api.marketplace.novo-genai.com/v1") # Use getenv
    MODEL_NAME = args.model_name
    MAX_TOKENS = args.max_new_tokens # Max tokens for the API response (OpenAI uses 'max_tokens')

    # --- Script Configuration ---
    csv_data_directory = "./data"
    output_dir = Path(f"./output/eval/{args.model_name}") # Ensure this directory exists
    output_filename = f"{MODEL_NAME}_results.jsonl" # Use model name in filename

    batch_size = 1 # batch size per device for API calls
    batches_to_print = 0 # How many batches to print for debugging

    # --- Helper Function to Parse Answer ---
    def extract_answer(generated_text):
        # Keep this function as is, assuming the API model follows the <answer> tag format
        if not isinstance(generated_text, str): # Add safety check
            return None
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Refined check for robustness
            if args.task == "COT":    
                if "upregulated" in answer: return "upregulated"
                if "downregulated" in answer: return "downregulated"
                # Handle "not differentially expressed" and "no"
                if "not differentially expressed" in answer: return "not differentially expressed"
                if "use gene enrichment libraries" in answer: return "use gene enrichment libraries"
            elif args.task == "direct_prediction":
                
                # Parse upregulated genes
                upregulated_pattern = r"Upregulated:\s*(\[.*?\])"
                upregulated_match = re.search(upregulated_pattern, answer)
                
                # Parse downregulated genes  
                downregulated_pattern = r"Downregulated:\s*(\[.*?\])"
                downregulated_match = re.search(downregulated_pattern, answer)
                
                try:
                    if upregulated_match:
                        upregulated_genes = eval(upregulated_match.group(1))  # Parse the list string
                    else:
                        upregulated_genes = []
                        
                    if downregulated_match:
                        downregulated_genes = eval(downregulated_match.group(1))  # Parse the list string
                    else:
                        downregulated_genes = []
                        
                    answer_content = {
                        "upregulated": upregulated_genes,
                        "downregulated": downregulated_genes
                    }
                except (SyntaxError, NameError):
                    # If parsing fails, set to None to indicate couldn't reliably recover
                    answer_content = None
                return answer_content
        # Return None if no match or none of the keywords found
        return None

    # --- Main Execution ---

    # --- Initialize Accelerator ---
    accelerator = Accelerator()
    device = accelerator.device # device might not be strictly needed for API calls but accelerator uses it
    accelerator.print(f"Process {accelerator.process_index} using device: {device}")

    # --- Initialize ChatOpenAI Client ---
    # Check for placeholder values specifically
    if API_KEY == "sk-7lN2yHKbX5NWkbzjFU0faQ":
        accelerator.print("Warning: Using placeholder MARKETPLACE_API_KEY. Please set the environment variable.")
        # Decide if you want to exit or proceed with a potential error later
        # exit(1)
    if BASE_URL == "https://api.marketplace.novo-genai.com/v1":
        accelerator.print("Warning: Using placeholder MARKETPLACE_BASE_URL. Please set the environment variable.")
        # exit(1)
    if not API_KEY:
        accelerator.print("Error: MARKETPLACE_API_KEY is not set.")
        exit(1)
    if not BASE_URL:
        accelerator.print("Error: MARKETPLACE_BASE_URL is not set.")
        exit(1)


    try:
        llm = ChatOpenAI(
            model=,
            api_key="sk-7lN2yHKbX5NWkbzjFU0faQ",
            base_url=BASE_URL,
            max_tokens=MAX_TOKENS, # Use max_tokens for OpenAI API
            max_retries=3,
            request_timeout=120,
        )
        if accelerator.is_main_process:
            print(f"ChatOpenAI client initialized for model: {MODEL_NAME} at {BASE_URL}")
    except Exception as e:
        accelerator.print(f"Error initializing ChatOpenAI: {e}")
        exit(1)


    # --- Load Test Dataset ---
    # Using main_process_first can prevent potential race conditions if
    # the dataset constructor writes cache files, although often unnecessary.

    if args.AUROC_stage == "dir":
        print("args.AUROC_stage == dir")
        prompt_mode = "o3_direction_test"
    else:
        prompt_mode = "o3_test"
    
    test_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, split="test", prompt_mode=prompt_mode)


    def collate_fn(batch):
        prompts = [item.get('prompt', None) for item in batch]
        
        # MODIFICATION HERE:
        # Decide what to put in 'solutions' based on the task
        if args.task == "direct_prediction":
            # For direct_prediction, we need the dictionary of gene lists
            solutions = [item.get('raw_solution_lists', None) for item in batch]
        else: # For single_gene_prediction, or other tasks, assume solution_text is fine
            solutions = [item.get('solution_text', None) for item in batch] # Or whatever 'single_gene_prediction' expects

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
        num_workers=1,
        pin_memory=False 
    )

    # --- Prepare DataLoader only ---
    # Model is not prepared as it's an API client
    test_dataloader = accelerator.prepare(test_dataloader)
    accelerator.print("DataLoader prepared.")

    # --- Run Inference and Evaluate ---
    total_correct_local = 0
    total_evaluated_local = 0
        # --- NEW: Local accumulators for other metrics ---
    local_sum_tp_up, local_sum_num_pred_up, local_sum_num_true_up = 0, 0, 0
    local_sum_tp_down, local_sum_num_pred_down, local_sum_num_true_down = 0, 0, 0
    local_f1_scores_up, local_f1_scores_down = [], []
    # --- END NEW ---

    results_local = []
    samples_printed_count = 0 # Keep track for debug printing limit

    accelerator.print(f"\nStarting distributed evaluation using {MODEL_NAME} API...")
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, total=len(test_dataloader),file=sys.stdout)

    # No torch.no_grad() needed for API calls
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
                progress_bar.set_postfix({"api_errors": progress_bar.postfix.get("api_errors", 0) + num_failed})
            time.sleep(2) # Add a small delay before retrying next batch after an error
            continue # Skip to the next batch

        # --- Compare generated answers with solutions locally ---
        for i, gen_text in enumerate(generated_texts):
            # Ensure index is valid for metadata lists
            if i >= len(solutions):
                accelerator.print(f"Warning: Index mismatch at batch {batch_idx}, item {i}. Generated text index out of bounds for solutions. Skipping result.")
                continue
                
            extracted_answer = extract_answer(gen_text)
            correct_solution = solutions[i] 
            is_correct_strict = False 

            # Initialize detailed metrics storage for this sample
            upregulated_metrics_results = None
            downregulated_metrics_results = None

                        # --- START DEBUG PRINTS ---
            # Only print for the first few batches/items to avoid flooding logs
            should_print_debug = accelerator.is_local_main_process

            if should_print_debug and args.task == "direct_prediction":
                accelerator.print(f"\n--- DEBUG START (Rank {accelerator.process_index}, Batch {batch_idx}, Item {i}) ---")
                accelerator.print(f"Generated Text: {gen_text}") # Print first 300 chars
                accelerator.print(f"Extracted Answer: {extracted_answer}")
                accelerator.print(f"Correct Solution Type: {type(correct_solution)}")
                accelerator.print(f"Correct Solution Value: {correct_solution}")
            # --- END DEBUG PRINTS ---
            is_correct_strict = False 


            if args.task == "single_gene_prediction":
                is_correct_strict = (extracted_answer == correct_solution)
            
            elif args.task == "direct_prediction":
                set_pred_up, set_pred_down = set(), set()
                set_true_up, set_true_down = set(), set()

                if extracted_answer is not None and isinstance(extracted_answer, dict):
                    # ... (logic to populate set_pred_up, set_pred_down) ...
                    pred_up_list = extracted_answer.get("upregulated", [])
                    pred_down_list = extracted_answer.get("downregulated", [])
                    if isinstance(pred_up_list, list) and all(isinstance(g, str) for g in pred_up_list):
                        set_pred_up = set(gene.strip().lower() for gene in pred_up_list if gene.strip())
                    if isinstance(pred_down_list, list) and all(isinstance(g, str) for g in pred_down_list):
                        set_pred_down = set(gene.strip().lower() for gene in pred_down_list if gene.strip())
                
                if isinstance(correct_solution, dict):
                    # ... (logic to populate set_true_up, set_true_down) ...
                    true_up_list = correct_solution.get("upregulated", [])
                    true_down_list = correct_solution.get("downregulated", [])
                    if isinstance(true_up_list, list) and all(isinstance(g, str) for g in true_up_list):
                        set_true_up = set(gene.strip().lower() for gene in true_up_list if gene.strip())
                    if isinstance(true_down_list, list) and all(isinstance(g, str) for g in true_down_list):
                        set_true_down = set(gene.strip().lower() for gene in true_down_list if gene.strip())
                
                upregulated_metrics_results = calculate_set_performance_metrics(set_pred_up, set_true_up)
                downregulated_metrics_results = calculate_set_performance_metrics(set_pred_down, set_true_down)
                
                if upregulated_metrics_results["f1"] == 1.0 and downregulated_metrics_results["f1"] == 1.0:
                    is_correct_strict = True
                
                # --- NEW: Accumulate for local detailed metrics ---
                local_sum_tp_up += upregulated_metrics_results["tp"]
                local_sum_num_pred_up += upregulated_metrics_results["num_pred"]
                local_sum_num_true_up += upregulated_metrics_results["num_true"]
                local_f1_scores_up.append(upregulated_metrics_results["f1"])

                local_sum_tp_down += downregulated_metrics_results["tp"]
                local_sum_num_pred_down += downregulated_metrics_results["num_pred"]
                local_sum_num_true_down += downregulated_metrics_results["num_true"]
                local_f1_scores_down.append(downregulated_metrics_results["f1"])
                # --- END NEW ---

            if is_correct_strict: # This `is_correct` refers to the strict, both-lists-perfect match
                total_correct_local += 1
            total_evaluated_local += 1 # Samples for which we got a response and tried to evaluate

            user_prompt_content = "Prompt content unavailable"
            # (Safely access prompt content - your existing logic for this)
            if i < len(prompts) and prompts[i] and isinstance(prompts[i], list) and len(prompts[i]) > 0:
                user_msg = next((msg for msg in reversed(prompts[i]) if isinstance(msg, (HumanMessage, dict)) and ( (isinstance(msg, HumanMessage) or msg.get('role') == 'user')) ), None)
                if user_msg:
                    if isinstance(user_msg, HumanMessage): user_prompt_content = user_msg.content
                    elif isinstance(user_msg, dict): user_prompt_content = user_msg.get('content', 'Prompt content unavailable')

            current_result = {
                "user_prompt": user_prompt_content,
                "pert": perts[i] if i < len(perts) else "N/A",
                "gene": genes[i] if i < len(genes) else "N/A",
                "cell_type": cell_types[i] if i < len(cell_types) else "N/A",
                "generated_text": gen_text,
                "extracted_answer": extracted_answer,
                "correct_solution": correct_solution,
                "is_correct_strict": is_correct_strict,
                "api_error": False
            }
            if args.task == "direct_prediction":
                current_result["upregulated_metrics"] = upregulated_metrics_results
                current_result["downregulated_metrics"] = downregulated_metrics_results
            results_local.append(current_result)


        # Update progress bar postfix with running accuracy (local to process)
        if accelerator.is_local_main_process and total_evaluated_local > 0: # and (batch_idx % 10 == 0 or batch_idx == len(test_dataloader) - 1): # Example: update every 10 batches
            postfix_metrics = {}
            local_strict_acc = (total_correct_local / total_evaluated_local) * 100
            postfix_metrics["StrictAcc"] = f"{local_strict_acc:.1f}%"

            if args.task == "direct_prediction" and local_sum_num_true_up > 0 : # Check to avoid division by zero if no direct_prediction samples yet
                # Local Micro F1 Up
                local_micro_p_up = local_sum_tp_up / local_sum_num_pred_up if local_sum_num_pred_up > 0 else (1.0 if local_sum_tp_up == 0 and local_sum_num_true_up == 0 else 0.0)
                local_micro_r_up = local_sum_tp_up / local_sum_num_true_up if local_sum_num_true_up > 0 else (1.0 if local_sum_tp_up == 0 and local_sum_num_pred_up == 0 else 0.0)
                if local_micro_p_up + local_micro_r_up == 0:
                    local_micro_f1_up = 0.0
                    if local_sum_num_pred_up == 0 and local_sum_num_true_up == 0 : local_micro_f1_up = 1.0
                else:
                    local_micro_f1_up = 2 * (local_micro_p_up * local_micro_r_up) / (local_micro_p_up + local_micro_r_up)
                postfix_metrics["μF1↑"] = f"{local_micro_f1_up:.2f}" # Micro F1 Upregulated

                # Local Macro F1 Up
                local_macro_f1_up = sum(local_f1_scores_up) / len(local_f1_scores_up) if local_f1_scores_up else 0.0
                postfix_metrics["MF1↑"] = f"{local_macro_f1_up:.2f}" # Macro F1 Upregulated
            
            if args.task == "direct_prediction" and local_sum_num_true_down > 0:
                 # Local Micro F1 Down
                local_micro_p_down = local_sum_tp_down / local_sum_num_pred_down if local_sum_num_pred_down > 0 else (1.0 if local_sum_tp_down == 0 and local_sum_num_true_down == 0 else 0.0)
                local_micro_r_down = local_sum_tp_down / local_sum_num_true_down if local_sum_num_true_down > 0 else (1.0 if local_sum_tp_down == 0 and local_sum_num_pred_down == 0 else 0.0)
                if local_micro_p_down + local_micro_r_down == 0:
                    local_micro_f1_down = 0.0
                    if local_sum_num_pred_down == 0 and local_sum_num_true_down == 0 : local_micro_f1_down = 1.0
                else:
                    local_micro_f1_down = 2 * (local_micro_p_down * local_micro_r_down) / (local_micro_p_down + local_micro_r_down)
                postfix_metrics["μF1↓"] = f"{local_micro_f1_down:.2f}" # Micro F1 Downregulated

                # Local Macro F1 Down
                local_macro_f1_down = sum(local_f1_scores_down) / len(local_f1_scores_down) if local_f1_scores_down else 0.0
                postfix_metrics["MF1↓"] = f"{local_macro_f1_down:.2f}" # Macro F1 Downregulated

            progress_bar.set_postfix(postfix_metrics)


    # --- Gather Results Across All Processes ---
    accelerator.print("Evaluation loop finished. Gathering results...")

    # First gather the simple metrics which are just numbers
    correct_tensor = torch.tensor(total_correct_local, device=accelerator.device)
    evaluated_tensor = torch.tensor(total_evaluated_local, device=accelerator.device)
    total_correct_gathered = accelerator.reduce(correct_tensor, reduction="sum")
    total_evaluated_gathered = accelerator.reduce(evaluated_tensor, reduction="sum")

    # Now use chunking for the complex results gathering
    CHUNK_SIZE = 100  # Adjust based on your data size
    all_chunks_local = [results_local[i:i + CHUNK_SIZE] 
                    for i in range(0, len(results_local), CHUNK_SIZE)]

    accelerator.print(f"Gathering detailed results in {len(all_chunks_local)} chunks...")
    all_results = []
    for chunk_idx, chunk in enumerate(all_chunks_local):
        accelerator.print(f"Gathering chunk {chunk_idx+1}/{len(all_chunks_local)}...")
        gathered_chunk = gather_object(chunk)
        if accelerator.is_main_process:
            if isinstance(gathered_chunk, list):
                for process_results in gathered_chunk:
                    if isinstance(process_results, list):
                        all_results.extend(process_results)
        # Add synchronization point
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"Gathered and assembled {len(all_results)} total results")


     # --- Process and Save Results (only on main process) ---
    if accelerator.is_main_process:
        print("\n--- Post-processing on Main Process ---")
        total_correct_strict_gathered = total_correct_gathered.item() # Samples where both lists were perfect
        total_evaluated_gathered = total_evaluated_gathered.item()    # Samples evaluated (API call succeeded)

        # The 'all_results' variable from the previous step in main() should be the one passed
        # from gather_object(results_local). It will be a flat list of dictionaries.
        # Let's assume the gathered_chunk logic successfully populated `all_results`
        # For clarity, if you have a chunking mechanism, ensure `all_results` is the final flattened list.
        # If no chunking, then `all_results = gather_object(results_local)` would be used directly here.

        # Let's refine the gathering logic based on your provided snippet:
        # The snippet initializes `all_results = []` then tries to fill it from `gathered_chunk`
        # This implies `gathered_chunk` from the previous loop is the actual list of lists.
        
        # Corrected Flattening based on your gather_object(chunk) loop:
        # `all_results` should be the list that was populated by extending `gathered_chunk` elements.
        # No need to re-initialize `all_results = []` here if it's already populated correctly.
        
        # Assuming `all_results` is now a flat list of all result dictionaries from all processes.
        
        print(f"Total items in all_results (before API error filtering): {len(all_results)}")
        
        # Filter out results with API errors for metric calculation, but count them.
        api_error_count = sum(1 for r in all_results if r.get('api_error', False))
        results_for_metrics = [r for r in all_results if not r.get('api_error', False)]
        
        print(f"Total API errors encountered across all processes: {api_error_count}")
        print(f"Total valid results for metric calculation: {len(results_for_metrics)}")

        # `total_evaluated_gathered` should match `len(results_for_metrics)` if no other filtering happened.
        # `total_correct_strict_gathered` is the sum of `is_correct_strict` flags from valid results.
        
        # --- Strict Overall Accuracy ---
        # This is the accuracy of getting BOTH lists entirely correct.
        if total_evaluated_gathered > 0:
            strict_accuracy_global = (total_correct_strict_gathered / total_evaluated_gathered) * 100
            print("\n--- Evaluation Summary ---")
            print(f"Model: {MODEL_NAME}")
            print(f"Total Samples Evaluated (API Succeeded): {total_evaluated_gathered}")
            print(f"Samples with Both Lists Perfectly Correct: {total_correct_strict_gathered}")
            print(f"Strict Overall Accuracy (Both lists perfect): {strict_accuracy_global:.2f}%")
        else:
            print("\nNo samples were successfully processed for strict accuracy (all API calls failed or dataset empty).")

        # --- Detailed Set Performance for "direct_prediction" task ---
        if args.task == "direct_prediction" and results_for_metrics: # Use the filtered list
            # Accumulators for micro-averages (overall TP, FP, FN, counts)
            total_tp_up, total_num_pred_up, total_num_true_up = 0, 0, 0
            total_tp_down, total_num_pred_down, total_num_true_down = 0, 0, 0
            
            # Lists for macro-averages (collect per-sample F1 scores)
            f1_scores_up, f1_scores_down = [], []
            
            num_valid_samples_for_set_metrics = 0 # Should be len(results_for_metrics) if all have metrics

            for res_item in results_for_metrics:
                # No need to check 'api_error' again as we're iterating over `results_for_metrics`
                
                up_m = res_item.get("upregulated_metrics")
                down_m = res_item.get("downregulated_metrics")

                if up_m and down_m: # Ensure metrics were calculated and present
                    num_valid_samples_for_set_metrics += 1
                    
                    total_tp_up += up_m["tp"]
                    total_num_pred_up += up_m["num_pred"]
                    total_num_true_up += up_m["num_true"]
                    
                    total_tp_down += down_m["tp"]
                    total_num_pred_down += down_m["num_pred"]
                    total_num_true_down += down_m["num_true"]
                    
                    f1_scores_up.append(up_m["f1"])
                    f1_scores_down.append(down_m["f1"])
                # else: # Optional: Log if a non-API error sample is missing metrics
                #     print(f"Warning: Sample without API error is missing direct_prediction metrics: {res_item.get('user_prompt', 'N/A')[:50]}")


            if num_valid_samples_for_set_metrics > 0:
                print(f"\n--- Detailed Set Performance (Direct Prediction, based on {num_valid_samples_for_set_metrics} valid samples) ---")

                # --- Upregulated Genes ---
                print("\nUpregulated Genes:")
                micro_p_up = total_tp_up / total_num_pred_up if total_num_pred_up > 0 else (1.0 if total_tp_up == 0 and total_num_true_up == 0 else 0.0)
                micro_r_up = total_tp_up / total_num_true_up if total_num_true_up > 0 else (1.0 if total_tp_up == 0 and total_num_pred_up == 0 else 0.0)
                if micro_p_up + micro_r_up == 0:
                    micro_f1_up = 0.0
                    if total_num_pred_up == 0 and total_num_true_up == 0 : micro_f1_up = 1.0 # Both pred and true were empty overall
                else:
                    micro_f1_up = 2 * (micro_p_up * micro_r_up) / (micro_p_up + micro_r_up)


                print(f"  Overall Precision (Micro): {micro_p_up*100:.2f}% ({total_tp_up}/{total_num_pred_up})")
                print(f"  Overall Recall (Micro):    {micro_r_up*100:.2f}% ({total_tp_up}/{total_num_true_up})")
                print(f"  Overall F1-Score (Micro):  {micro_f1_up:.4f}")
                
                macro_f1_up_avg = sum(f1_scores_up) / len(f1_scores_up) if f1_scores_up else 0.0
                print(f"  Average F1-Score (Macro):  {macro_f1_up_avg:.4f}")

                # --- Downregulated Genes ---
                print("\nDownregulated Genes:")
                micro_p_down = total_tp_down / total_num_pred_down if total_num_pred_down > 0 else (1.0 if total_tp_down == 0 and total_num_true_down == 0 else 0.0)
                micro_r_down = total_tp_down / total_num_true_down if total_num_true_down > 0 else (1.0 if total_tp_down == 0 and total_num_pred_down == 0 else 0.0)
                if micro_p_down + micro_r_down == 0:
                    micro_f1_down = 0.0
                    if total_num_pred_down == 0 and total_num_true_down == 0 : micro_f1_down = 1.0 # Both pred and true were empty overall
                else:
                    micro_f1_down = 2 * (micro_p_down * micro_r_down) / (micro_p_down + micro_r_down)

                print(f"  Overall Precision (Micro): {micro_p_down*100:.2f}% ({total_tp_down}/{total_num_pred_down})")
                print(f"  Overall Recall (Micro):    {micro_r_down*100:.2f}% ({total_tp_down}/{total_num_true_down})")
                print(f"  Overall F1-Score (Micro):  {micro_f1_down:.4f}")

                macro_f1_down_avg = sum(f1_scores_down) / len(f1_scores_down) if f1_scores_down else 0.0
                print(f"  Average F1-Score (Macro):  {macro_f1_down_avg:.4f}")

            elif args.task == "direct_prediction": # And results_for_metrics was empty or had no metrics
                 print("\nNo valid samples with direct prediction metrics found to aggregate detailed set performance.")
        
        # --- Save Results ---
        if all_results: # Save all results, including those with API errors (they have api_error=True flag)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename

            print(f"\nSaving {len(all_results)} detailed results (including API errors) to {output_path}...")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for result_item in all_results:
                        f.write(json.dumps(result_item) + '\n')
                print(f"Results saved successfully to {output_path}")
            except TypeError as e:
                print(f"Error saving results to JSON: {e}. Check if any non-serializable data types are in the results.")
            except Exception as e:
                print(f"Error saving results to file: {e}")
        else:
            print("No detailed results were gathered to save.")

    # Ensure all processes finish before exiting
    accelerator.wait_for_everyone()
    accelerator.print(f"Rank {accelerator.process_index} finished script.")