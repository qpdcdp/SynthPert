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
from src.data import DiffExpressionDataset
from src.utils.enrichr_old import find_pathways, generate_prompt

# Import Accelerator
from accelerate import Accelerator
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO)
# Reduce httpx verbosity unless debugging API calls specifically
logging.getLogger('httpx').setLevel(logging.WARNING) # Changed from DEBUG to WARNING

def main(args):
    # --- Configuration ---
    # --- OpenAI API Configuration ---
    # *** IMPORTANT: Set these environment variables or replace the placeholders ***
    API_KEY = os.getenv("MARKETPLACE_API_KEY", "sk-7lN2yHKbX5NWkbzjFU0faQ") # Use getenv for flexibility
    BASE_URL = os.getenv("MARKETPLACE_BASE_URL", "https://api.marketplace.novo-genai.com/v1") # Use getenv
    MODEL_NAME = args.model_name
    LLM_TEMPERATURE = 1 # Adjust as needed
    MAX_TOKENS = args.max_new_tokens # Max tokens for the API response (OpenAI uses 'max_tokens')

    # --- Script Configuration ---
    csv_data_directory = "/workspace/PertRL/data"
    output_dir = Path(f"/workspace/PertRL/output/eval/{args.model_name}/{args.context}") # Ensure this directory exists
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
            answer = match.group(1).strip().lower()
            # Refined check for robustness
        if "upregulated" in answer: return "upregulated"
        
        if "downregulated" in answer: return "downregulated"
        # Handle "not differentially expressed" and "no"
        if "not differentially expressed" in answer: return "not differentially expressed"
    
        if "use gene enrichment libraries" in answer: return "use gene enrichment libraries"

        return None # Explicitly return None if none of the keywords found

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
            model=MODEL_NAME,
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
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            print("Loading test dataset definition...")
        try:
            test_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, split="test", prompt_mode="o3_test", context="none", tool=args.tool)
        except (ValueError, RuntimeError, FileNotFoundError) as e: # Catch more specific errors
            accelerator.print(f"Error loading dataset from {csv_data_directory}: {e}")
            exit(1) # Exit if dataset loading fails

        if accelerator.is_main_process:
            print(f"Full test dataset size: {len(test_dataset)} samples.")
            if len(test_dataset) == 0:
                print("Warning: Test dataset is empty.")
                # Decide if you want to exit if the dataset is empty
                # exit(1)


    # --- Collate function (Simplified for API) ---
    def collate_fn(batch):
        # Extract necessary fields, use .get for safety with metadata
        # Expects 'prompt' key to contain the list of messages for the chat API
        prompts = [item.get('prompt', None) for item in batch]
        solutions = [item.get('solution', None) for item in batch]
        perts = [item.get('pert', None) for item in batch]
        genes = [item.get('gene', None) for item in batch]
        cell_types = [item.get('cell_type', None) for item in batch]

        # Filter out None prompts if any occurred
        # Ensure prompt is a list (chat format) and not empty
        valid_indices = [i for i, p in enumerate(prompts) if p is not None and isinstance(p, list) and len(p) > 0]
        if len(valid_indices) != len(prompts):
            warnings.warn(f"Batch contained {len(prompts) - len(valid_indices)} items with missing, invalid, or empty 'prompt' key.")
            # Filter other lists accordingly
            prompts = [prompts[i] for i in valid_indices]
            solutions = [solutions[i] for i in valid_indices]
            perts = [perts[i] for i in valid_indices]
            genes = [genes[i] for i in valid_indices]
            cell_types = [cell_types[i] for i in valid_indices]
        
        if not prompts: # If batch becomes empty after filtering
            return None

        # No tokenization needed here for API calls
        # Just return the data needed for the API call and evaluation

        return {
            "prompts": prompts, # This should be the list of message lists
            "solutions": solutions,
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
        num_workers=1, # Increased workers slightly
        pin_memory=False # pin_memory not needed/useful here
    )

    # --- Prepare DataLoader only ---
    # Model is not prepared as it's an API client
    test_dataloader = accelerator.prepare(test_dataloader)
    accelerator.print("DataLoader prepared.")

    # --- Run Inference and Evaluate ---
    total_correct_local = 0
    total_evaluated_local = 0
    results_local = []
    samples_printed_count = 0
    api_error_count_local = 0 # Track API errors locally

    accelerator.print(f"\nStarting distributed evaluation using {MODEL_NAME} API (processing items individually)...") # Modified message
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Evaluating", disable=not accelerator.is_local_main_process, total=len(test_dataloader), file=sys.stdout)

    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            accelerator.print(f"Skipping empty batch {batch_idx} on Rank {accelerator.process_index}")
            continue

        prompts_list_of_dicts = batch['prompts'] # List of lists of dictionaries
        solutions = batch['solutions']
        perts = batch['perts']
        genes = batch['genes']
        cell_types = batch['cell_types']

        if not prompts_list_of_dicts:
            accelerator.print(f"Skipping batch {batch_idx} due to empty prompts list on Rank {accelerator.process_index}")
            continue

        # --- Process items within the batch individually ---
        for i in range(len(prompts_list_of_dicts)):
            # Get data for the current item
            item_prompt_dicts = prompts_list_of_dicts[i]
            item_solution = solutions[i]
            item_pert = perts[i]
            item_gene = genes[i]
            item_cell_type = cell_types[i]

            # --- Variables to store final results for this item ---
            final_generated_text = "N/A"
            final_extracted_answer = None
            final_user_prompt_str = "Prompt unavailable" # Store the prompt string that led to the final answer
            api_error_occurred = False
            used_enrichment = False
            system_prompt_used = None
            original_user_query = None

            try:
                # Convert dict messages to LangChain message objects for invoke
                first_pass_messages = []
                for msg_dict in item_prompt_dicts:
                    role = msg_dict.get('role')
                    content = msg_dict.get('content')
                    if not role or not content: continue
                    if role == 'system':
                        first_pass_messages.append(SystemMessage(content=content))
                        system_prompt_used = content # Store system prompt
                    elif role == 'user':
                        first_pass_messages.append(HumanMessage(content=content))
                        if original_user_query is None: original_user_query = content # Store first user query
                    elif role == 'assistant':
                        # This shouldn't happen in the test prompt, but handle if needed
                        first_pass_messages.append(AIMessage(content=content))

                if not first_pass_messages or original_user_query is None:
                    raise ValueError("Could not construct valid messages or find user query for API call.")
                if system_prompt_used is None: # Add default if missing
                    first_pass_messages.insert(0, SystemMessage(content="Default system prompt needed if not provided")) # Or use your actual default
                    system_prompt_used = "Default system prompt used"

                final_user_prompt_str = original_user_query # Default final prompt is the original one

                # --- First API Call ---
                accelerator.print(f"Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Making first API call...") if samples_printed_count < batches_to_print else None
                response1 = llm.invoke(first_pass_messages)
                first_generated_text = response1.content if isinstance(response1, AIMessage) else str(response1)

                # --- Parse First Response ---
                extracted_answer1 = extract_answer(first_generated_text)

                # --- Conditional Enrichment Step ---

                if extracted_answer1 == "use gene enrichment libraries":
                    used_enrichment = True
                    accelerator.print(f"Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Enrichment requested.")

                    # Prepare for enrichment (keep this part)
                    pert_gene = item_pert
                    target_gene = item_gene
                    pert_gene_symbol = str(pert_gene[:-3]) if isinstance(pert_gene, str) and pert_gene.endswith("_KD") else str(pert_gene)
                    target_gene_symbol = str(target_gene)

                    if pert_gene_symbol and target_gene_symbol:
                        sample_genes = [pert_gene_symbol, target_gene_symbol]
                        accelerator.print(f"Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Running find_pathways for {sample_genes}")
                        pathways = find_pathways(sample_genes)

                        if pathways:
                            # --- Create the *content* for the new user message ---
                            enrichr_output = generate_prompt(
                                pathways=pathways,
                                sample_genes=sample_genes
                            )

                            enrichr_prompt = (
                                f"Here is the gene enrichment analysis data from Enrichr related to the {target_gene} gene and the "
                                f"knocked down {pert_gene} gene, which may inform your prediction:\n\n{enrichr_output}"
                            )
                            second_turn_content = (
                                "\n\n--- Task ---\n" # Separator for clarity
                                "Considering the original query, the initial reasoning provided (which led to requesting enrichment), AND the new pathway information above, "
                                "please provide a final, refined reasoning process within <think> </think> tags and a definitive prediction in the <answer> tag. "
                                "Your final prediction in the <answer> tag must be one of: 'upregulated', 'downregulated', or 'not differentially expressed'. "
                                "Do not output any other options (like 'use gene enrichment libraries' or 'I do not know') in the answer tag for this final step."
                            )
                            # --- Construct the full message history for the second call ---
                            # Start with the messages from the first call
                            # Use deepcopy to avoid modifying the original list if looping/retrying
                            second_pass_messages = copy.deepcopy(first_pass_messages)
                            # Append the AI's first response
                            if isinstance(response1, AIMessage): # Ensure response1 is an AIMessage
                                second_pass_messages.append(response1)
                            else:
                                # Handle cases where response1 might not be AIMessage (though it should be from invoke)
                                second_pass_messages.append(AIMessage(content=str(response1)))

                            # Append the new user message for this turn
                            second_pass_messages.append(HumanMessage(content=second_turn_content))

                            # Store the string content of the *last* user message for logging/critic later if needed
                            # Note: The 'user_prompt' for the critic might now just be this second turn content,
                            # or you might want to reconstruct the full conversation string for the critic.
                            # Let's keep `final_user_prompt_str` as the content of the last Human turn for now.
                            final_user_prompt_str = second_turn_content


                            # --- Second API Call ---
                            accelerator.print(f"Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Making second API call with history...")
                            # Pass the full message list
                            response2 = llm.invoke(second_pass_messages)
                            final_generated_text = response2.content if isinstance(response2, AIMessage) else str(response2)
                            # Parse the *final* answer from the *second* response
                            final_extracted_answer = extract_answer(final_generated_text)
                        else:
                            # Enrichment failed or returned no pathways
                            accelerator.print(f"Warning: Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Enrichment requested but no pathways found. Treating as failed generation.")
                            final_generated_text = first_generated_text + "\n\n[Enrichment Step Failed: No Pathways Found]"
                            final_extracted_answer = None # Mark as failed parsing
                            api_error_occurred = True # Treat enrichment failure like an error for scoring? Or handle differently? Let's flag it.
                    else:
                        # Couldn't get gene symbols
                        accelerator.print(f"Warning: Rank {accelerator.process_index} Batch {batch_idx} Item {i}: Enrichment requested but failed to get gene symbols. Treating as failed generation.")
                        final_generated_text = first_generated_text + "\n\n[Enrichment Step Failed: Invalid Genes]"
                        final_extracted_answer = None
                        api_error_occurred = True
                else:
                    # No enrichment needed, use results from the first call
                    final_generated_text = first_generated_text
                    final_extracted_answer = extracted_answer1
                
                    final_user_prompt_str = original_user_query

                # --- Final Comparison ---
                is_correct = (final_extracted_answer is not None and final_extracted_answer == item_solution)

            except Exception as e:
                # Catch errors during API calls or processing for this item
                accelerator.print(f"\nError processing item on Rank {accelerator.process_index}, Batch {batch_idx}, Item Index {i}: {e}")
                api_error_occurred = True
                is_correct = False
                final_generated_text = f"Error: {e}"
                final_extracted_answer = None
                api_error_count_local += 1 # Increment local API error count
                # Attempt to get original user query if available before error
                if original_user_query: final_user_prompt_str = original_user_query[:200] + "... (Error Occurred)"
                else: final_user_prompt_str = "Prompt unavailable (Error Occurred)"
                time.sleep(1) # Small delay after an item error

            # --- Store results for this item ---
            # Only count towards accuracy if no API/processing error occurred for the item
            if not api_error_occurred:
                total_evaluated_local += 1
                if is_correct:
                    total_correct_local += 1

            # Append result regardless of error for complete logging
            results_local.append({
                "user_prompt": final_user_prompt_str, # Store the prompt that led to the final answer
                "pert": item_pert,
                "gene": item_gene,
                "cell_type": item_cell_type,
                "generated_text": final_generated_text, # Store the final generated text
                "extracted_answer": final_extracted_answer, # Store the final extracted answer
                "correct_solution": item_solution,
                "is_correct": is_correct,
                "used_enrichment": used_enrichment, # Add flag
                "api_error": api_error_occurred # Flag if an error happened for this item
            })
            samples_printed_count +=1


        # --- Update progress bar postfix after processing batch ---
        if accelerator.is_local_main_process:
            local_acc = (total_correct_local / total_evaluated_local * 100) if total_evaluated_local > 0 else 0
            progress_bar.set_postfix({
                "Local Acc": f"{local_acc:.2f}% ({total_correct_local}/{total_evaluated_local})",
                "Errors": api_error_count_local
            })


    # --- Gather Results Across All Processes ---
    # (Keep the gathering logic using accelerator.reduce and gather_object exactly as it was)
    accelerator.print(f"Rank {accelerator.process_index} finished evaluation loop. Correct: {total_correct_local}, Evaluated: {total_evaluated_local}, API Errors: {api_error_count_local}. Gathering results...")

    correct_tensor = torch.tensor(total_correct_local, device=accelerator.device)
    evaluated_tensor = torch.tensor(total_evaluated_local, device=accelerator.device)
    api_error_tensor = torch.tensor(api_error_count_local, device=accelerator.device) # Gather API errors too

    total_correct_gathered = accelerator.reduce(correct_tensor, reduction="sum")
    total_evaluated_gathered = accelerator.reduce(evaluated_tensor, reduction="sum")
    total_api_errors_gathered = accelerator.reduce(api_error_tensor, reduction="sum") # Reduce API errors

    accelerator.print(f"Rank {accelerator.process_index} gathering detailed results objects...")
    gathered_results_list = gather_object(results_local)
    accelerator.print(f"Rank {accelerator.process_index} finished gathering.")


    # --- Process and Save Results (only on main process) ---
    # (Update the final summary section to use the gathered API error count)
    if accelerator.is_main_process:
        print("\n--- Post-processing on Main Process ---")
        total_correct = total_correct_gathered.item()
        total_evaluated = total_evaluated_gathered.item() # This count excludes items with API/processing errors
        total_api_errors = total_api_errors_gathered.item() # Get the total API error count

        print(f"Total correct predictions gathered (summed): {total_correct}")
        print(f"Total evaluated samples gathered (summed, excludes item errors): {total_evaluated}")
        print(f"Total API/processing errors encountered for items: {total_api_errors}")

        # Flatten and Save Detailed Results
        all_results = []
        if gathered_results_list and isinstance(gathered_results_list, list):
            # ... (rest of the flattening logic remains the same) ...
            print(f"Gathered results list contains {len(gathered_results_list)} sub-lists.")
            if all(isinstance(sublist, list) for sublist in gathered_results_list):
                for process_results in gathered_results_list:
                    all_results.extend(process_results)
                print(f"Gathered and flattened {len(all_results)} detailed results.")

                # Verify total_api_errors matches count from list (optional check)
                api_error_check = sum(1 for r in all_results if r.get('api_error', False))
                print(f"Total API errors counted from results list: {api_error_check}") # Should match total_api_errors

                # Calculate accuracy excluding item errors
                if total_evaluated > 0:
                    accuracy = (total_correct / total_evaluated) * 100
                    print("\n--- Evaluation Complete ---")
                    print(f"Model: {MODEL_NAME}")
                    print(f"Total Samples Successfully Processed (Global, excludes item errors): {total_evaluated}")
                    print(f"Correct Predictions (Global): {total_correct}")
                    print(f"Accuracy (excluding item errors): {accuracy:.2f}%")
                else:
                    print("\nNo samples were successfully processed globally (excluding item errors).")

                # Optionally calculate accuracy including item errors as incorrect
                total_attempts = len(all_results) # Total attempts is the total number of results logged
                if total_attempts > 0:
                    # Correct count remains the same
                    overall_accuracy = (total_correct / total_attempts) * 100
                    print(f"Total Attempts Logged (including errors): {total_attempts}")
                    print(f"Overall Accuracy (treating errors as incorrect): {overall_accuracy:.2f}%")
                else:
                    print("No results were generated or logged at all.")

                # ... (rest of the saving logic remains the same) ...
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / output_filename
                print(f"Saving {len(all_results)} detailed results to {output_path}...")
                # ... (try/except block for saving) ...
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for result_item in all_results:
                            f.write(json.dumps(result_item) + '\n')
                    print(f"Results saved successfully to {output_path}")
                except TypeError as e:
                    print(f"Error saving results to JSON: {e}. Check for non-serializable data.")
                except Exception as e:
                    print(f"Error saving results to file: {e}")

            else:
                print("Warning: gathered_results_list structure unexpected. Skipping save.")
        else:
            print(f"No detailed results were gathered or format is unexpected (Type: {type(gathered_results_list)}). Total evaluated: {total_evaluated}, Total correct: {total_correct}")

    # Ensure all processes finish before exiting
    accelerator.wait_for_everyone()
    accelerator.print(f"Rank {accelerator.process_index} finished script.")