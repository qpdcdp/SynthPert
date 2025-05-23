import csv
import pandas as pd
import numpy as np
import re
import os
import sys
import random
from typing import Optional, List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# --- Import your existing Enrichr helpers ---
from src.utils.enrichr_old import find_pathways, generate_prompt


from src.data import DiffExpressionDataset


# Disable excessive logging from libraries like httpx
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
    "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
    "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
    "gene interactions, and cell-specific context. "
    "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', 'not differentially expressed', OR if you determine pathway analysis is essential and lacking, state 'use gene enrichment libraries'."
    "Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed / use gene enrichment libraries] </answer>"
)


def extract_think_and_answer(generated_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts content within <think>...</think> and <answer>...</answer> tags."""
    think_match = re.search(r"<think>(.*?)</think>", generated_text, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)

    thinking_content = think_match.group(1).strip() if think_match else None
    answer_content = None
    raw_answer_text = None # Store the raw text for logging

    if answer_match: # Check if the tag was found at all
        raw_answer_text = answer_match.group(1).strip().lower() # Store raw answer text
        # Check for specific keywords
        if "upregulated" in raw_answer_text: answer_content = "upregulated"
        elif "downregulated" in raw_answer_text: answer_content = "downregulated"
        elif "not differentially expressed" in raw_answer_text: answer_content = "not differentially expressed"
        elif "use gene enrichment libraries" in raw_answer_text: answer_content = "use gene enrichment libraries" # Check for the enrichment keyword
        else: answer_content = None # Could not reliably recover any known answer

    # --- Logging adjustments ---
    log_prefix = f"Raw Text: '{generated_text[:100]}...' "
    if thinking_content is None and answer_content is None:
         logging.debug(log_prefix + "Failed to parse BOTH <think> and <answer> tags.")
         return None, None

    if thinking_content is None:
        logging.debug(log_prefix + "Failed to parse <think> tag.")
        return None, answer_content

    if answer_content is None:
         if answer_match:
             logging.debug(log_prefix + f"Could not parse known keywords from <answer> tag content: '{raw_answer_text}'.")
         else:
             logging.debug(log_prefix + "Could not parse <answer> tag.")
         return thinking_content, None

    return thinking_content, answer_content


def extract_critic_evaluation(critic_response_text: str) -> Tuple[Optional[str], Optional[str]]:
    """ Extracts evaluation and reasoning from critic using regex. """
    eval_match = re.search(r"<evaluation>(.*?)</evaluation>", critic_response_text, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r"<reasoning>(.*?)</reasoning>", critic_response_text, re.IGNORECASE | re.DOTALL)

    evaluation = eval_match.group(1).strip().lower() if eval_match else None
    critic_reasoning = reason_match.group(1).strip() if reason_match else None

    valid_evaluations = ["excellent", "good", "average", "bad", "terrible"]
    if evaluation not in valid_evaluations:
        logging.warning(f"Critic evaluation unclear. Expected one of {valid_evaluations}. Got: '{evaluation}'. Raw: {critic_response_text[:100]}...")
        return None, critic_reasoning
    return evaluation, critic_reasoning


def create_critic_prompt(user_query: str, generated_thinking: str) -> List[SystemMessage | HumanMessage]:
    """ Creates the prompt for the critic model using Langchain schema. """
    system_prompt = """You are an expert molecular biologist acting as a critic.
Your task is to evaluate the reasoning process of another AI model that was asked to predict gene expression changes based on a perturbation.
Focus *only* on the quality, logical flow, and biological relevance of the provided reasoning (<think> block). Do not judge the final answer, only the steps taken to reach it.
Is the reasoning sound? Does it mention relevant and correct biological concepts (pathways, mechanisms, functions)? Does it logically connect the perturbation to the gene in the given cell type context? Did it appropriately use pathway information if provided?
Output your evaluation *only* in the following format choosing a single value for the evaluation:
<evaluation>[excellent/good/average/bad/terrible]</evaluation>
<reasoning>Provide a brief justification for your evaluation here. Explain why the reasoning is excellent, good, average, bad, or terrible.</reasoning>"""

    critic_input = f"""Original User Query (potentially augmented with pathway info):
{user_query}

AI's Final Reasoning (<think> block):
{generated_thinking}

Critique Task:
Evaluate the AI's reasoning based on the criteria mentioned in the system prompt. Output your evaluation and justification in the specified format (<evaluation>...</evaluation><reasoning>...</reasoning>).
"""
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=critic_input)
    ]


# --- Data Processing Function ---
def process_data_item(
    data_item: Dict[str, Any],
    llm_generator: ChatOpenAI,
    llm_critic: ChatOpenAI,
    default_system_prompt: str
    ) -> Optional[Dict[str, Any]]:
    """
    Processes a single data item, potentially using Enrichr via imported helpers.
    (Full docstring omitted for brevity, see previous version)
    """
    item_id_log = f"pert={data_item.get('pert', '?')},gene={data_item.get('gene', '?')}"
    used_enrichment = False

    try:
        # --- 1. Initial Generator Step ---
        generator_chat_prompt_list = data_item['prompt']
        true_solution = data_item['solution']

        generator_messages = []
        user_query = None
        system_prompt_used = default_system_prompt
        for msg in generator_chat_prompt_list:
            role, content = msg.get('role'), msg.get('content')
            if not role or not content: continue
            if role == 'system':
                generator_messages.append(SystemMessage(content=content))
                system_prompt_used = content
            elif role == 'user':
                generator_messages.append(HumanMessage(content=content))
                if user_query is None: user_query = content
            elif role == 'assistant':
                generator_messages.append(AIMessage(content=content))
            else: logging.warning(f"({item_id_log}) Unknown role: {role}")

        if not any(isinstance(m, HumanMessage) for m in generator_messages) or user_query is None:
             logging.warning(f"({item_id_log}) No user query found.")
             return None
        if not any(isinstance(m, SystemMessage) for m in generator_messages):
             logging.warning(f"({item_id_log}) No system prompt found. Adding default.")
             generator_messages.insert(0, SystemMessage(content=default_system_prompt))

        # --- Invoke Generator LLM (First Pass) ---
        try:
            logging.debug(f"({item_id_log}) Invoking generator (pass 1)...")
            generator_response = llm_generator.invoke(generator_messages)
            generated_text = generator_response.content
        except Exception as gen_e:
            logging.error(f"({item_id_log}) Generator LLM invocation (pass 1) failed: {gen_e}")
            return None

        # Parse Generator Output (First Pass)
        thinking, extracted_answer = extract_think_and_answer(generated_text)
        logging.debug(f"({item_id_log}) Parsed (pass 1): thinking={thinking is not None}, answer='{extracted_answer}'")

        # --- 2. Conditional Enrichment and Re-Prompt Step ---
        if extracted_answer == "use gene enrichment libraries":
            used_enrichment = True
            logging.info(f"({item_id_log}) Generator requested enrichment.")

            pert_gene = data_item.get("pert")
            target_gene = data_item.get("gene")
            pert_gene_symbol = str(pert_gene[:-3]) if isinstance(pert_gene, str) and pert_gene.endswith("_KD") else str(pert_gene)
            target_gene_symbol = str(target_gene)

            if pert_gene_symbol and target_gene_symbol: # Basic check
                sample_genes = [pert_gene_symbol, target_gene_symbol]
                logging.debug(f"({item_id_log}) Genes for enrichment: {sample_genes}")

                 # --- Call your imported find_pathways ---
                try:
                    pathways = find_pathways(sample_genes)
                     # Assuming find_pathways returns None or {} on failure/no results
                    if pathways:
                        logging.info(f"({item_id_log}) Found pathways via enrichr_old: {list(pathways.keys()) if isinstance(pathways, dict) else 'data received'}") # Log keys or confirmation

                         # --- Call your imported generate_prompt ---
                         # Adapt arguments if your generate_prompt signature is different!
                         # Assuming it needs the original query context, pathways, and genes.

                        enrichr_output = generate_prompt(user_query, pathways, sample_genes)
                        
                        enrichr_prompt = (
                            f"Here is the gene enrichment analysis data from Enrichr related to the {target_gene} gene and the "
                            f"knocked down {pert_gene} gene, which may inform your prediction:\n\n{enrichr_output}"
                        )
                        final_instruction = (
                            "\n\n--- Task ---\n" # Separator for clarity
                            "Considering the original query, the initial reasoning provided (which led to requesting enrichment), AND the new pathway information above, "
                            "please provide a final, refined reasoning process within <think> </think> tags and a definitive prediction in the <answer> tag. "
                            "Your final prediction in the <answer> tag must be one of: 'upregulated', 'downregulated', or 'not differentially expressed'. "
                            "Do not output any other options (like 'use gene enrichment libraries' or 'I do not know') in the answer tag for this final step."
                        )
                     
                        user_query = f"{enrichr_prompt}{final_instruction}"
                        new_user_prompt = (
                            f"--- Original User Query ---\n{user_query}\n\n" # Add header for clarity
                            f"--- Initial AI Response (Requesting Enrichment) ---\n{generated_text}\n" # Include the first full response
                            f"{enrichr_prompt}" # Pathway info already includes its own header
                            f"{final_instruction}" # Final task instruction includes its own header
                        )


                         # --- Invoke Generator LLM (Second Pass) ---
                        second_pass_messages = [
                            SystemMessage(content=system_prompt_used),
                            HumanMessage(content=new_user_prompt)
                        ]
                        try:
                            logging.debug(f"({item_id_log}) Invoking generator (pass 2 with pathways)...")
                            generator_response = llm_generator.invoke(second_pass_messages)
                            generated_text = generator_response.content
                            thinking, extracted_answer = extract_think_and_answer(generated_text) # Re-parse
                            logging.debug(f"({item_id_log}) Parsed (pass 2): thinking={thinking is not None}, answer='{extracted_answer}'")
                            user_query = new_user_prompt # Update query for critic context

                        except Exception as gen_e_2:
                            logging.error(f"({item_id_log}) Generator LLM invocation (pass 2) failed: {gen_e_2}")
                            return None
                    else:
                        logging.warning(f"({item_id_log}) Enrichment suggested but find_pathways from enrichr_old returned no pathways. Skipping.")
                        return None # Skip if enrichment failed

                except Exception as enrich_e:
                    logging.error(f"({item_id_log}) Error during enrichr_old call (find_pathways or generate_prompt): {enrich_e}", exc_info=True)
                    return None # Skip if enrichment functions raise error

            else:
                 logging.warning(f"({item_id_log}) Enrichment suggested, but could not extract valid gene symbols from pert='{pert_gene}' and gene='{target_gene}'. Skipping enrichment.")
                 return None

        # --- End Conditional Enrichment ---

        # --- 3. Critic Step ---
        if thinking is None:
            logging.warning(f"({item_id_log}) Failed to extract final <think> block. Skipping critique.")
            return None

        final_valid_answers = ["upregulated", "downregulated", "not differentially expressed"]
        if extracted_answer not in final_valid_answers:
            logging.warning(f"({item_id_log}) Final generator answer ('{extracted_answer}') is not one of {final_valid_answers}. Skipping.")
            return None

        # Prepare and Invoke Critic
        critic_prompt_messages = create_critic_prompt(user_query, thinking)
        try:
            logging.debug(f"({item_id_log}) Invoking critic...")
            critic_response = llm_critic.invoke(critic_prompt_messages)
            critic_response_text = critic_response.content
        except Exception as crit_e:
            logging.error(f"({item_id_log}) Critic LLM invocation failed: {crit_e}")
            return None

        # Parse Critic Output
        critic_evaluation, critic_reasoning = extract_critic_evaluation(critic_response_text)
        if critic_evaluation is None:
            logging.warning(f"({item_id_log}) Failed to parse critic evaluation. Skipping.")
            return None

        # --- 4. Decision and Formatting Step ---
        is_approved = critic_evaluation in ["good", "excellent"]
        answer_matches = extracted_answer == true_solution
        logging.debug(f"({item_id_log}) Evaluation: approved={is_approved} ('{critic_evaluation}'), answer_match={answer_matches} (gen='{extracted_answer}', true='{true_solution}')")

        if is_approved and answer_matches:
            return {
                "pert": data_item.get('pert'),
                "gene": data_item.get('gene'),
                "cell_type": data_item.get('cell_type'),
                "true_answer": true_solution,
                "system_prompt": system_prompt_used,
                "user_prompt": user_query, # Final user query
                "assistant_response": generated_text, # Final assistant response
                "generated_thinking": thinking, # Final thinking
                "generated_answer": extracted_answer, # Final answer
                "used_enrichment": used_enrichment,
                "critic_evaluation": critic_evaluation,
                "critic_reasoning": critic_reasoning if critic_reasoning else "N/A"
            }
        else:
            discard_reason = []
            if not is_approved: discard_reason.append(f"Critic eval: '{critic_evaluation}'")
            if not answer_matches: discard_reason.append(f"Answer mismatch (Gen: '{extracted_answer}', True: '{true_solution}')")
            logging.debug(f"({item_id_log}) Discarded: {', '.join(discard_reason)}")
            return None

    except Exception as e:
        logging.error(f"({item_id_log}) Unexpected error in process_data_item: {str(e)}", exc_info=True)
        return None

# (Keep the main function and argument parsing as they were in the previous response,
# ensuring the call to process_data_item remains correct:
# executor.submit(process_data_item, data, llm_generator, llm_critic, DEFAULT_SYSTEM_PROMPT)
# )

# --- Example main execution structure (keep from previous version) ---
def main(args):
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)
    API_KEY = os.getenv("NOVO_GENAI_API_KEY", "sk-7lN2yHKbX5NWkbzjFU0faQ") # Use env var
    BASE_URL = "https://api.marketplace.novo-genai.com/v1"
    GENERATOR_MODEL_NAME = args.generator_model_name
    CRITIC_MODEL_NAME = args.critic_model_name

    llm_generator = ChatOpenAI(model=GENERATOR_MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, max_retries=3, request_timeout=120, max_tokens=4096)
    llm_critic = ChatOpenAI(model=CRITIC_MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, max_retries=3, request_timeout=120)
    logging.info(f"Initialized Generator: {GENERATOR_MODEL_NAME}, Critic: {CRITIC_MODEL_NAME}")

    CSV_DATA_DIRECTORY = args.csv_data_directory
    OUTPUT_CSV_FILE = os.path.join(args.output_dir, f"synthetic_data_{args.generator_model_name}_{args.synth_data_script}_tool-enrichr.csv")
    TRAIN_SUBSET_PERCENTAGE = args.train_subset_fraction
    MAX_WORKERS = args.num_workers

    logging.info("--- Starting Synthetic Data Generation with Critic & Imported Enrichment ---")

    # 1. Load Dataset
    logging.info(f"Loading train split from: {CSV_DATA_DIRECTORY}")
    try:
        full_train_dataset = DiffExpressionDataset(csv_dir=CSV_DATA_DIRECTORY, split="train", prompt_mode="o3_synth_data_with_ans", tool="enrichr")
        logging.info(f"Full train dataset loaded with {len(full_train_dataset)} samples.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}", exc_info=True)
        sys.exit(1)

    # 2. Create Subset
    dataset_size = len(full_train_dataset)
    if dataset_size == 0: logging.error("Dataset empty."); exit(1)
    subset_size = max(1, int(dataset_size * TRAIN_SUBSET_PERCENTAGE)) if dataset_size > 0 else 0
    logging.info(f"Creating subset of {subset_size} samples ({TRAIN_SUBSET_PERCENTAGE*100:.1f}%).")
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]
    dataset_subset_to_process = [full_train_dataset[i] for i in subset_indices]

    # 3. Process Data in Parallel
    approved_count = 0
    processed_count = 0
    enrichment_triggered_count = 0
    output_fieldnames = [
        "pert", "gene", "cell_type", "true_answer", "system_prompt",
        "user_prompt", "assistant_response", "generated_thinking",
        "generated_answer", "used_enrichment", "critic_evaluation", "critic_reasoning"
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

    try:
        with open(OUTPUT_CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output_fieldnames)
            writer.writeheader()
            logging.info(f"Opened {OUTPUT_CSV_FILE} for writing.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_data_item, data, llm_generator, llm_critic, DEFAULT_SYSTEM_PROMPT): data
                    for data in dataset_subset_to_process
                }
                logging.info(f"Submitted {len(futures)} tasks.")

                progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Generating & Critiquing", file=sys.stdout)
                for future in progress_bar:
                    processed_count += 1
                    try:
                        result = future.result()
                        if result is not None:
                            writer.writerow(result)
                            approved_count += 1
                            if result.get("used_enrichment"): enrichment_triggered_count += 1
                    except Exception as e:
                         item_data = futures[future]
                         item_id_log = f"pert={item_data.get('pert', '?')},gene={item_data.get('gene', '?')}"
                         logging.error(f"({item_id_log}) Error processing future: {e}", exc_info=True)

                    approval_rate = (approved_count / processed_count * 100) if processed_count else 0
                    progress_bar.set_postfix({"Approved": approved_count, "Enriched": enrichment_triggered_count, "Rate": f"{approval_rate:.2f}%"})
                    # progress_bar.update(0) # Might not be needed depending on tqdm version/env

            file.flush()
            logging.info("Processing complete.")

    except IOError as e:
        logging.error(f"Error with CSV file {OUTPUT_CSV_FILE}: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during parallel processing: {e}")
        exit(1)

    # 4. Final Summary
    logging.info("--- Summary (Critic + Imported Enrichment) ---")
    logging.info(f"Processed {processed_count}/{len(dataset_subset_to_process)} items.")
    logging.info(f"Triggered enrichment path for {enrichment_triggered_count} items.")
    logging.info(f"Generated {approved_count} approved examples.")
    if processed_count > 0: logging.info(f"Approval Rate: {(approved_count / processed_count * 100):.2f}%")
    logging.info(f"Output saved to: {OUTPUT_CSV_FILE}")
    logging.info("--- Script Finished ---")

