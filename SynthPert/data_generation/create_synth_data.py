import csv
import pandas as pd
import numpy as np
import re
import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage # Use schema for messages

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from src.data import DiffExpressionDataset



# Disable excessive logging from libraries like httpx
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(args):

    # --- Configuration ---
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)

    # --- LLM Configuration ---

    BASE_URL = "https://api.marketplace.novo-genai.com/v1"
    MODEL_NAME = 'openai_o4_mini' # Using the model from the coworker's example for generation

    # Initialize LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key="sk-7lN2yHKbX5NWkbzjFU0faQ",
        base_url=BASE_URL,
        max_retries=3, # Add some retries for transient network issues
        request_timeout=120 # Increase timeout for potentially longer generations
    )
    logging.info(f"Initialized ChatOpenAI model: {MODEL_NAME}")


    # --- Data Configuration ---
    # Directory containing your dataset CSVs
    # Directory containing your dataset CSVs
    CSV_DATA_DIRECTORY = args.csv_data_directory
    # Output file for the correctly generated synthetic data
    split_name = "default" if args.test_split_cell_lines == "none" else args.test_split_cell_lines
    prompt_type = "generator_lemmas" if args.generator_lemmas else "default_prompt"
    OUTPUT_DIR = args.output_dir + f"{args.generator_model_name}/{args.synth_data_script}/{split_name}_split/"
    OUTPUT_CSV_FILE = OUTPUT_DIR + f"{prompt_type}.csv"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    # Percentage of training data to use for generation
    TRAIN_SUBSET_PERCENTAGE = args.train_subset_fraction
    # Maximum number of parallel workers for API calls
    MAX_WORKERS = 40 # Adjust based on your API rate limits and system resources

    # --- Helper Function to Parse LLM Output ---
    def extract_think_and_answer(generated_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts content within <think>...</think> and <answer>...</answer> tags.
        Returns (thinking_content, answer_content).
        Returns (None, None) if tags are not found or format is incorrect.
        """
        think_match = re.search(r"<think>(.*?)</think>", generated_text, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)

        thinking_content = think_match.group(1).strip() if think_match else None
        answer_content = answer_match.group(1).strip().lower() if answer_match else None

        # Basic validation for the answer format
        if answer_content not in ["upregulated", "downregulated", "not differentially expressed"]:
            # Try to recover if the answer text contains the keywords
            if answer_match: # Check if the tag was found at all
                raw_answer = answer_match.group(1).strip().lower()
                if "upregulated" in raw_answer: answer_content = "upregulated"
                elif "downregulated" in raw_answer: answer_content = "downregulated"
                elif "not differentially expressed" in raw_answer: answer_content = "not differentially expressed"
                else: answer_content = None # Could not reliably recover
            else:
                answer_content = None # Tag wasn't even found

        if thinking_content is None or answer_content is None:
            logging.warning(f"Could not parse think/answer tags reliably from: '{generated_text}'")
            return None, None # Indicate parsing failure

        return thinking_content, answer_content

    # --- Data Processing Function ---
    def process_data_item(data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single data item: generates response, parses, evaluates, and formats for saving.

        Args:
            data_item: A dictionary obtained from DiffExpressionDataset.__getitem__.
                    Expected keys: 'prompt' (list of chat messages), 'solution' (str: "upregulated", "downregulated", "not differentially expressed").
                    Also expects original data like 'pert', 'gene', 'cell_type' if available in data_item.

        Returns:
            A dictionary containing the data to be saved if the generation was correct, otherwise None.
        """
        try:
            # The prompt is already formatted as a list of dicts by DiffExpressionDataset
            chat_prompt = data_item['prompt']
            true_solution = data_item['solution']

            # Convert prompt format if necessary for LangChain (expects BaseMessage objects)
            messages = []
            for msg in chat_prompt:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant': # Include if your format has assistant messages
                    messages.append(AIMessage(content=msg['content']))
                else:
                    logging.warning(f"Unknown role encountered in prompt: {msg['role']}")
                    # Handle appropriately, maybe skip or raise error

            if not messages:
                logging.warning("Empty message list generated from prompt.")
                return None

            # Invoke the LLM
            response = llm.invoke(messages)
            generated_text = response.content

            # Parse the generated text
            thinking, extracted_answer = extract_think_and_answer(generated_text)

            # Evaluate if the extracted answer is correct
            if extracted_answer is not None and extracted_answer == true_solution:
                # If correct, prepare the data for saving
                # We want to save the original user query and the full generated text for SFT
                user_query = ""
                for msg in chat_prompt:
                    if msg['role'] == 'user':
                        user_query = msg['content']
                        break # Assuming one user message per item based on DiffExpressionDataset

                if not user_query:
                    logging.warning("Could not find user query in the prompt structure.")
                    # Fallback or handle as needed, here we save the full prompt as JSON
                    user_query = json.dumps(chat_prompt)


                # Add original data fields if they exist in the input data_item
                # These were commented out in __getitem__ but might be useful context
                pert = data_item.get('pert', 'N/A')
                gene = data_item.get('gene', 'N/A')
                cell_type = data_item.get('cell_type', 'N/A')

                # Return a dictionary structured for the CSV output
                return {
                    "pert": pert,
                    "gene": gene,
                    "cell_type": cell_type,
                    "user_query": user_query, # The input question/instruction
                    "true_answer": true_solution, # Ground truth
                    "assistant_response": generated_text, # The full <think>...</think><answer>...</answer> string
                    "generated_thinking": thinking, # Parsed thinking part
                    "generated_answer": extracted_answer # Parsed answer part (which matches true_answer here)
                }
            else:
                # Log incorrect or unparsable responses if needed for debugging
                # logging.debug(f"Incorrect or unparsable response. True: {true_solution}, Got: {extracted_answer}, Raw: {generated_text[:100]}...")
                return None # Indicate failure/incorrectness

        except Exception as e:
            # Log the error with more context if possible
            item_id = f"pert={data_item.get('pert', '?')},gene={data_item.get('gene', '?')}"
            logging.error(f"Error processing data item ({item_id}): {str(e)}", exc_info=False) # Set exc_info=True for full traceback
            return None

    # --- Main Execution ---
    logging.info("--- Starting Synthetic Data Generation ---")

    # 1. Load Dataset
    try:
        logging.info(f"Loading train split from: {CSV_DATA_DIRECTORY}")
        prompt_mode = "default_with_biologist_lemmas" if args.generator_lemmas else  "o3_synth_data"
        full_train_dataset = DiffExpressionDataset(csv_dir=CSV_DATA_DIRECTORY, split="train", prompt_mode=prompt_mode)
        logging.info(f"Full train dataset loaded with {len(full_train_dataset)} samples.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        exit(1)

    # 2. Create Subset
    dataset_size = len(full_train_dataset)
    subset_size = int(dataset_size * TRAIN_SUBSET_PERCENTAGE)
    if subset_size == 0 and dataset_size > 0:
        subset_size = 1 # Ensure at least one sample if percentage is very small
        logging.warning(f"Subset percentage resulted in 0 samples, using 1 sample instead.")
    elif subset_size == 0:
        logging.error("Dataset is empty or subset size is zero. Cannot proceed.")
        exit(1)


    logging.info(f"Creating a random subset of {subset_size} samples ({TRAIN_SUBSET_PERCENTAGE*100:.1f}% of train).")
    # Get all indices, shuffle them, and take the first `subset_size`
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices) # Shuffle indices in place (uses the random seed set earlier)
    subset_indices = all_indices[:subset_size]

    # Create the actual subset to iterate over
    # We can just get the items by index directly when needed
    dataset_subset_to_process = [full_train_dataset[i] for i in subset_indices]
    logging.info(f"Subset created. Processing {len(dataset_subset_to_process)} items.")


    # 3. Process Data in Parallel and Write to CSV
    correct_count = 0
    processed_count = 0
    output_fieldnames = [
        "pert", "gene", "cell_type", # Original context
        "user_query",                 # Input prompt to the model
        "true_answer",                # Ground truth label
        "assistant_response",             # Full generated output from LLM (incl. tags)
        "generated_thinking",         # Extracted thinking part
        "generated_answer"            # Extracted answer part (should match true_answer)
    ]

    try:
        with open(OUTPUT_CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output_fieldnames)
            writer.writeheader()
            logging.info(f"Opened {OUTPUT_CSV_FILE} for writing.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                futures = {executor.submit(process_data_item, data): data for data in dataset_subset_to_process}
                logging.info(f"Submitted {len(futures)} tasks to the thread pool.")

                # Process completed tasks
                progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Generating Data")
                for future in progress_bar:
                    processed_count += 1
                    try:
                        result = future.result() # Get result from the future
                        if result is not None: # Check if the function returned data (i.e., was correct)
                            writer.writerow(result)
                            correct_count += 1
                            # Optional: Flush occasionally to save progress
                            # if correct_count % 50 == 0:
                            #     file.flush()
                        # Update progress bar description
                        accuracy = (correct_count / processed_count * 100) if processed_count > 0 else 0
                        progress_bar.set_postfix({"Correct": correct_count, "Processed": processed_count, "Acc": f"{accuracy:.2f}%"})

                    except Exception as e:
                        # This catches errors *retrieving* the result, process_data_item handles internal errors
                        logging.error(f"Error retrieving result from future: {str(e)}")

            file.flush() # Final flush
            logging.info("Processing complete.")

    except IOError as e:
        logging.error(f"Error opening or writing to CSV file {OUTPUT_CSV_FILE}: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during parallel processing: {e}")
        exit(1)

    # 4. Final Summary
    logging.info("--- Synthetic Data Generation Summary ---")
    logging.info(f"Processed {processed_count} / {len(dataset_subset_to_process)} items from the subset.")
    logging.info(f"Generated {correct_count} correct examples.")
    if processed_count > 0:
        final_accuracy = (correct_count / processed_count) * 100
        logging.info(f"Generation Accuracy (on subset): {final_accuracy:.2f}%")
    logging.info(f"Correct examples saved to: {OUTPUT_CSV_FILE}")
    logging.info("--- Script Finished ---")