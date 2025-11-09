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

from src.data import DiffExpressionDataset, GeneRegulationListDataset, PerturbationPredictionDataset, MetabolicFluxDataset, SeahorseClusterDataset


# Disable excessive logging from libraries like httpx
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(args):
    # --- Configuration ---
    SEED = 88
    np.random.seed(SEED)
    random.seed(SEED)

    BASE_URL = "https://api.marketplace.novo-genai.com/v1"
    API_KEY = "xxxxx" # It's better to load this from an env var
    GENERATOR_MODEL_NAME = args.generator_model_name
    CRITIC_MODEL_NAME = args.critic_model_name


        # --- Data Configuration ---
    CSV_DATA_DIRECTORY = args.csv_data_directory
    # Path to the CSV file containing the synthetic dataset
    split_name = "default" if args.test_split_cell_lines == "none" else args.test_split_cell_lines
    prompt_type = "critic_lemmas" if args.critic_lemmas else "default_prompt"
    threshold = ','.join(args.critic_acceptance_threshold)
    OUTPUT_CSV_FILE = args.output_dir + f"{args.task}/{args.generator_model_name}_as_gen_{args.critic_model_name}_as_critic/{args.synth_data_script}/{split_name}_split_{prompt_type}_{threshold}_critic_threshold.csv"
    # Percentage of training data to use for generation
    TRAIN_SUBSET_PERCENTAGE = args.train_subset_fraction
    # Maximum number of parallel workers for API calls
    MAX_WORKERS = args.num_workers # Adjust based on your API rate limits and system resources


    def sanitize_for_csv(text: str) -> str:
        """
        Sanitizes a string to prevent it from breaking CSV formatting.
        - Replaces newline characters with a space.
        - Removes isolated double-quote characters.
        """
        if not isinstance(text, str):
            return text
        # Replace newlines and carriage returns with a space
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Remove double quotes. A more advanced approach might be to replace them
        # with single quotes or escape them, but removing them is safest.
        text = text.replace('"', '')
        return text
    
    def extract_think_and_answer(generated_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts content within <think>...</think> and <answer>...</answer> tags."""
        think_match = re.search(r"<think>(.*?)</think>", generated_text, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)

        thinking_content = think_match.group(1).strip() if think_match else None
        answer_content = None
        raw_answer_text = None


        #TODO: Add more robust parsing for the answer content
        if answer_match and args.task == "single_gene_prediction": # Check if the tag was found at all
            raw_answer = answer_match.group(1).strip().lower()
            if "upregulated" in raw_answer: answer_content = "upregulated"
            elif "downregulated" in raw_answer: answer_content = "downregulated"
            elif "not differentially expressed" in raw_answer: answer_content = "not differentially expressed"
            else: answer_content = None # Could not reliably recover
        
        elif answer_match and args.task == "direct_prediction" and args.list_format == "default":
            raw_answer = answer_match.group(1).strip()
            
            # Parse upregulated genes
            upregulated_pattern = r"Upregulated:\s*(\[.*?\])"
            upregulated_match = re.search(upregulated_pattern, raw_answer)
            
            # Parse downregulated genes  
            downregulated_pattern = r"Downregulated:\s*(\[.*?\])"
            downregulated_match = re.search(downregulated_pattern, raw_answer)
            
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
        
        elif answer_match and args.task == "direct_prediction" and args.list_format == "bullet_list":

            raw_answer = answer_match.group(1).strip()
        
            result = {'upregulated': [], 'downregulated': []}
            
            # Find upregulated genes section
            up_match = re.search(r'UPREGULATED_GENES:\s*(.*?)(?=DOWNREGULATED_GENES:|$)', 
                               raw_answer, re.DOTALL | re.IGNORECASE)
            
            # Find downregulated genes section
            down_match = re.search(r'DOWNREGULATED_GENES:\s*(.*?)$', 
                                 raw_answer, re.DOTALL | re.IGNORECASE)
            
            # Process upregulated genes if found
            if up_match:
                up_content = up_match.group(1).strip()
                if up_content and up_content.upper() != "NONE":
                    # Extract genes from bullet points
                    up_genes = re.findall(r'-\s*([^\n]+)', up_content)
                    result['upregulated'] = [gene.strip() for gene in up_genes]
            
            # Process downregulated genes if found
            if down_match:
                down_content = down_match.group(1).strip()
                if down_content and down_content.upper() != "NONE":
                    # Extract genes from bullet points
                    down_genes = re.findall(r'-\s*([^\n]+)', down_content)
                    result['downregulated'] = [gene.strip() for gene in down_genes]
            
            answer_content = result

        elif answer_match and args.task == "inverse_problem":
            raw_answer = answer_match.group(1).strip()
            # We can do a basic check to ensure it's not a complex sentence.
            # A simple gene name shouldn't have spaces, but some might have dashes.
            if raw_answer and ' ' not in raw_answer and len(raw_answer) < 50:
                answer_content = raw_answer
            else:
                answer_content = None

        elif answer_match and args.task == "metabolic_flux":
            raw_answer = answer_match.group(1).strip().lower()
            if "increased" in raw_answer: answer_content = "increased"
            elif "decreased" in raw_answer: answer_content = "decreased"
            elif "not changed" in raw_answer: answer_content = "not changed"
            else: answer_content = None

        elif answer_match and args.task == "seahorse_cluster":
            answer_content = answer_match.group(1).strip().lower()

        if thinking_content is None or answer_content is None:
            # Log only if *both* are missing, or if answer tag was found but unparsable
            if thinking_content is None:
                logging.debug(f"Could not parse <think> tag from: '{generated_text[:100]}...'")
            if answer_match and answer_content is None: # Answer tag found but content invalid
                logging.debug(f"Could not parse answer keywords from <answer> tag content: '{raw_answer_text}'")
            elif not answer_match: # Answer tag not found
                logging.debug(f"Could not parse <answer> tag from: '{generated_text[:100]}...'")
            # Return None only if thinking is missing, as reasoning is key for the critic
            if thinking_content is None:
                return None, None

        # Return thinking content even if answer is dubious, critic focuses on thinking
        return thinking_content, answer_content


    def extract_critic_evaluation(critic_response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts the evaluation (e.g., 'good' or 'bad') from the critic's response.
        Assumes critic responds with <evaluation>[excellent / good / average / bad / terrible]</evaluation> and optionally <reasoning>...</reasoning>.
        """
        reason_match = re.search(r"<reasoning>(.*?)</reasoning>", critic_response_text, re.IGNORECASE | re.DOTALL)
        eval_match = re.search(r"<evaluation>(.*?)</evaluation>", critic_response_text, re.IGNORECASE | re.DOTALL)

        critic_reasoning = reason_match.group(1).strip() if reason_match else None
        evaluation = eval_match.group(1).strip().lower() if eval_match else None

        # Validate evaluation
        if evaluation not in ["excellent","good","average","bad","terrible"]:
            logging.warning(f"Critic evaluation unclear or missing. Expected 'good' or 'bad'. Got: '{evaluation}'. Raw critic response: {critic_response_text[:100]}...")
            return None, critic_reasoning # Return reasoning if available, but mark evaluation as failed

        return evaluation, critic_reasoning


    # --- Critic Prompt Construction ---
    def create_critic_prompt(user_query: str, generated_thinking: str) -> List[Dict[str, str]]:
        """Creates the prompt for the critic model."""
        
        # --- Start with a base prompt ---
        system_prompt = """You are an expert molecular and cellular biology expert acting as a critic.
    Your task is to evaluate the reasoning process of another AI model.
    Focus *only* on the quality, logical flow, and biological soundness of the provided reasoning (<think> block). Do not judge the final answer, only the steps taken to reach it.
    Is the reasoning sound? Does it mention relevant and correct biological concepts?
    Output your evaluation *only* in the following format, choosing a single value for the evaluation:
    <reasoning> [Provide a brief justification for your evaluation here.] </reasoning>
    <evaluation> [excellent/good/average/bad/terrible] </evaluation>"""

        # --- Dynamically insert the task-specific context ---
        if args.task == "direct_prediction" or args.task == "single_gene_prediction":
            task_specific_instruction = (
                "The AI model was asked to predict **gene expression changes based on a known perturbation**.\n"
                "Evaluate whether the reasoning logically connects the perturbation to the predicted gene expression outcomes in the given cell type context."
            )
        elif args.task == "inverse_problem":
            task_specific_instruction = (
                "The AI model was asked to predict **the causative perturbation gene based on a known list of gene expression changes**.\n"
                "Evaluate whether the reasoning logically connects the observed gene expression changes back to the function of the predicted perturbation gene in the given cell type context."
            )
        elif args.task == "metabolic_flux":
            task_specific_instruction = (
                "The AI model was asked to predict **the metabolic flux changes based on a known perturbation**.\n"
                "Evaluate whether the reasoning logically connects the perturbation to the predicted metabolic flux outcomes in the given contexts."
            )
        elif args.task == "seahorse_cluster":
            task_specific_instruction = (
                "The AI model was asked to predict **the seahorse cluster based on a known perturbation**.\n"
                "Evaluate whether the reasoning logically connects the perturbation to the predicted seahorse cluster outcomes in the given contexts."
            )
        # Inject the specific instruction into the system prompt
        # A good place is right after the main task definition.
        lines = system_prompt.splitlines()
        lines.insert(2, task_specific_instruction) # Insert after the second line
        system_prompt = "\n".join(lines)
        
        lemmas_string = (
            "\n\nPlease consider the following biology specific lemmas to guide your reasoning:\n"
            "biology-specific \"lemmas\":\n"
            "1. Gene Function Lemma - What is the known molecular function of the perturbed gene product? (e.g., transcription factor, kinase, receptor, enzymes, etc.)\n"
            "2. Pathway membership Lemma – What are the molecular pathways to which the perturbed and target genes are part of?\n"
            "3. Gene perturbation phenotypic outcome Lemma – what is the known cell phenotypic outcome of either mutating, silencing, activating, or inhibiting the perturbed gene?\n"
            "4. Pathway Positioning Lemma – What is the functional role of the perturbed gene in the pathways is part of? Is the perturbed gene an activator, mediator or repressor m of any of the pathways the target gene is part of?\n"
            "5. Loci regulatory landscape Lemma – what is the regulatory landscape of the genetic loci where the target gene is located? Which transcription factor controls the target gene expression? Which cell stimuli affects the target gene expression?\n"
            "6. Direct Regulation Lemma - Is there evidence of direct gene-expression regulatory connection between the perturbed and target genes?\n"
            "7. Cell Type Context Lemma - How does the specific cell type influence the regulatory relationship between these genes?\n"
            #"8. Cell type treatment context Lemma - . Could the cellular conditions of the experiment affect the regulatory interaction betweeen the perturbed and target genes?Perturbation Mechanism Lemma - How does the specific perturbation method (CRISPR interference) affect gene expression?\n"
            "8. Analog perturbations modalities Lemma – Are there compounds or active pharmacological ingredients targeting the perturbed gene? What is the cellular response to these compounds?\n"
            "9. Temporal Dynamics Lemma - What are the expected timeframes for observing effects after perturbation?\n"
            "10. Secondary Effects Lemma - What indirect effects might occur through other intermediary genes or feedback loops?\n"
            "11. Conflicting Evidence Lemma - What contradictory evidence exists in the literature about this relationship?\n"
            "Each of these would be established with evidence and reasoning before proceeding to the final conclusion."
        )
        
        if args.critic_lemmas:
            system_prompt += lemmas_string
        critic_input = f"""Original User Query:
    {user_query}

    AI's Reasoning (<think> block):
    {generated_thinking}

    Critique Task:
    Evaluate the AI's reasoning based on the criteria mentioned in the system prompt. Output your evaluation and justification in the specified format (<evaluation>...</evaluation><reasoning>...</reasoning>).
    """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critic_input}
        ]

    # --- Data Processing Function ---
    def process_data_item(data_item: Dict[str, Any], 
                        generator_model_name: str, 
                        critic_model_name: str, 
                        base_url: str, 
                        api_key: str,
                        task: str,
                        list_format: str,
                        critic_acceptance_threshold: list,
                        ) -> Optional[Dict[str, Any]]:
        """
        Processes a single data item. CRITICAL: Initializes its own LLM clients
        to ensure thread safety.
        """
        # --- LLM Initialization (INSIDE the worker function) ---
        # Each thread will create its own instance.
        try:
            llm_generator = ChatOpenAI(
                model=generator_model_name,
                api_key="xxxxx",
                base_url=base_url,
                max_retries=3,
                request_timeout=120,
            )
            llm_critic = ChatOpenAI(
                model=critic_model_name,
                api_key="xxxxx",
                base_url=base_url,
                max_retries=3,
                request_timeout=120,
            )
        except Exception as e:
            logging.error(f"Failed to initialize LLM in worker thread: {e}")
            return None
        item_id_log = f"pert={data_item.get('pert', '?')},gene={data_item.get('gene', '?')}" # For logging context
        logging.info(f"({item_id_log}) Starting processing.")

        try:
            # --- 1. Generator Step ---
    
            generator_chat_prompt = data_item['prompt']
            if args.task == "single_gene_prediction":
                true_solution = data_item['solution'] # Keep ground truth for reference
            elif args.task == "direct_prediction":
                true_solution = data_item['raw_solution_lists']
            elif args.task == "inverse_problem":
                true_solution = data_item['raw_solution_pert']
            elif args.task == "metabolic_flux":
                true_solution = data_item['solution']
            elif args.task == "seahorse_cluster":
                true_solution = data_item['solution']
            # Convert prompt for LangChain
            generator_messages = []
            user_query = None
            
            print("----Generator Prompt----")

            for msg in generator_chat_prompt:
                role = msg.get('role')
                content = msg.get('content')
                if not role or not content: continue # Skip malformed messages
                if role == 'system':
                    generator_messages.append(SystemMessage(content=content))
                    print(f"[{role.upper()}]\n{content}\n")
                elif role == 'user':
                    generator_messages.append(HumanMessage(content=content))
                    if user_query is None: user_query = content # Capture first user message
                    print(f"[{role.upper()}]\n{content}\n")
                elif role == 'assistant':
                    generator_messages.append(AIMessage(content=content))
                    print(f"[{role.upper()}]\n{content}\n")
                else:
                    # This 'else' is only reached if the role is NOT system, user, or assistant
                    logging.warning(f"({item_id_log}) Unknown role in generator prompt: {role}")

            if not generator_messages or user_query is None:
                logging.warning(f"({item_id_log}) Empty message list or no user query found for generator.")
                return None

            # Invoke Generator LLM
            try:
                generator_response = llm_generator.invoke(generator_messages)
                generated_text = generator_response.content
            except Exception as gen_e:
                logging.error(f"({item_id_log}) Generator LLM invocation failed: {gen_e}")
                return None # Cannot proceed without generator output
            
            print("----Generator Output----")
            print(generated_text)
            # Parse Generator Output
            thinking, extracted_answer = extract_think_and_answer(generated_text)
            print("----Extracted Generator Thinking----")
            print(thinking)
            print("----Extracted Generator Answer----")
            print(extracted_answer)
            print("----End of Generator Output----")
            if thinking is None:
                # If reasoning cannot be extracted, we cannot critique it.
                logging.warning(f"({item_id_log}) Failed to extract <think> block from generator. Skipping critique.")
                # Log the problematic output for debugging
                logging.debug(f"({item_id_log}) Generator raw output (start): {generated_text[:200]}...")
                return None
        
            # --- 2. Critic Step ---
            # Prepare Critic Prompt
            critic_prompt_messages_list = create_critic_prompt(user_query, thinking)
            critic_prompt_langchain = []
            critic_prompt_text_for_log = "" # For saving to CSV
            for msg in critic_prompt_messages_list:
                role = msg.get('role')
                content = msg.get('content')
                critic_prompt_text_for_log += f"[{role.upper()}]\n{content}\n\n"
                if role == 'system': critic_prompt_langchain.append(SystemMessage(content=content))
                elif role == 'user': critic_prompt_langchain.append(HumanMessage(content=content))
                # Add assistant if needed for critic prompt structure

            # Invoke Critic LLM
            try:
                critic_response = llm_critic.invoke(critic_prompt_langchain)
                critic_response_text = critic_response.content
            except Exception as crit_e:
                logging.error(f"({item_id_log}) Critic LLM invocation failed: {crit_e}")
                return None # Cannot proceed without critic evaluation

            print("----Critic Output----")
            print(critic_response_text)


            # Parse Critic Output
            critic_evaluation, critic_reasoning = extract_critic_evaluation(critic_response_text)

            print("----Critic Parsed Eval-----")
            print(critic_evaluation)

            if critic_evaluation is None:
                # If critique is unclear/unparseable, we cannot determine if reasoning is good.
                logging.warning(f"({item_id_log}) Failed to parse critic evaluation ('good'/'bad'). Skipping.")
                return None
            print("----Generated Answer----")
            print(extracted_answer)
            print("----True Solution----")
            print(true_solution)
            print("----End----")
            # --- 3. Decision and Formatting Step ---
            print(f"args.critic_acceptance_threshold:{args.critic_acceptance_threshold}")
            print(f"critic_evaluation:{critic_evaluation}")
            
            is_approved = critic_evaluation in args.critic_acceptance_threshold
            print(f"is_approved:{is_approved}")
            answer_matches = extracted_answer.lower() == true_solution.lower()
            print(f"answer_matches:{answer_matches}")

            if is_approved and answer_matches:

                generated_text = sanitize_for_csv(generated_text)
                thinking = sanitize_for_csv(thinking)
                critic_reasoning = sanitize_for_csv(critic_reasoning) if critic_reasoning else "N/A"

                print("---- Approved ---")
                # *** Prepare dictionary for CSV row ***
                if args.task == "single_gene_prediction" or "metabolic_flux" or "seahorse_cluster":
                    return {
                        "pert": data_item['pert'],
                        "gene": data_item['gene'],
                        "cell_type": data_item['cell_type'],
                        "true_answer": true_solution,
                        "user_prompt": user_query,             # Store user prompt
                        "assistant_response": generated_text,  # Store full generated response
                        "generated_thinking": thinking,
                        "generated_answer": extracted_answer,
                        "critic_evaluation": critic_evaluation,
                        "critic_reasoning": critic_reasoning if critic_reasoning else "N/A"
                    }
                elif args.task == "direct_prediction":
                    return {
                        "pert": data_item['pert'],
                        "solution": data_item['solution'],
                        "raw_solution_lists": data_item['raw_solution_lists'],
                        "cell_type": data_item['cell_type'],
                        "true_answer": true_solution,
                        "user_prompt": user_query,             # Store user prompt
                        "assistant_response": generated_text,  # Store full generated response
                        "generated_thinking": thinking,
                        "generated_answer": extracted_answer,
                        "critic_evaluation": critic_evaluation,
                        "critic_reasoning": critic_reasoning if critic_reasoning else "N/A"
                    }    
                
                elif args.task == "inverse_problem":
                    return {
                        "cell_type": data_item['cell_type'],
                        "true_pert": true_solution, # The ground truth perturbation
                        "input_upregulated_genes": str(data_item['raw_input_lists']['upregulated']),
                        "input_downregulated_genes": str(data_item['raw_input_lists']['downregulated']),
                        "user_prompt": user_query,
                        "assistant_response": generated_text,
                        "generated_thinking": thinking,
                        "generated_answer": extracted_answer, # The model's predicted perturbation
                        "critic_evaluation": critic_evaluation,
                        "critic_reasoning": critic_reasoning if critic_reasoning else "N/A"
                    }
            else:
                if critic_evaluation != "good":
                    logging.debug(f"({item_id_log}) Discarded: Critic evaluation was '{critic_evaluation}'.")
                elif extracted_answer is None:
                    logging.debug(f"({item_id_log}) Discarded: Generator answer parsing failed. Raw generated text: {generated_text[:100]}...")
                elif extracted_answer != true_solution:
                    logging.debug(f"({item_id_log}) Discarded: Generator answer '{extracted_answer}' != True answer '{true_solution}'.")

                return None # Indicate failure/rejection by critic

        except Exception as e:
            logging.error(f"({item_id_log}) Unexpected error in process_data_item: {str(e)}", exc_info=False)
            return None

    # --- Main Execution ---
    
    logging.info("--- Starting Synthetic Data Generation with Critic ---")

    # 1. Load Dataset

    logging.info(f"Loading train split from: {CSV_DATA_DIRECTORY}")
    if args.task == "single_gene_prediction":
        full_train_dataset = DiffExpressionDataset(csv_dir=CSV_DATA_DIRECTORY, split="train", prompt_mode="o3_synth_data_with_ans", context = args.context, test_split_cell_lines = args.test_split_cell_lines)
    elif args.task == "direct_prediction":
        full_train_dataset = GeneRegulationListDataset(csv_dir=CSV_DATA_DIRECTORY, split="train", prompt_mode="synth_data_with_ans", list_format = args.list_format, test_split_cell_lines = args.test_split_cell_lines)
    elif args.task == "inverse_problem":
        full_train_dataset = PerturbationPredictionDataset(csv_dir=CSV_DATA_DIRECTORY, split="train", prompt_mode="synth_data_with_ans", test_split_cell_lines = args.test_split_cell_lines)
    elif args.task == "metabolic_flux":
        full_train_dataset = MetabolicFluxDataset(csv_path=CSV_DATA_DIRECTORY, split="train", prompt_mode="synth_data_with_ans", test_split_cell_lines = args.test_split_cell_lines)
    elif args.task == "seahorse_cluster":
        full_train_dataset = SeahorseClusterDataset(csv_path=CSV_DATA_DIRECTORY, split="train", prompt_mode="synth_data_with_ans")    
    
    logging.info(f"Full train dataset loaded with {len(full_train_dataset)} samples.")

    # 2. Create Subset
    dataset_size = len(full_train_dataset)
    if dataset_size == 0:
        logging.error("Dataset is empty. Cannot proceed.")
        exit(1)
    logging.info(f"Dataset_size: {dataset_size}")
    subset_size = int(dataset_size * TRAIN_SUBSET_PERCENTAGE)
    if subset_size == 0: subset_size = min(1, dataset_size) # Ensure at least one sample if possible

    logging.info(f"Creating a random subset of {subset_size} samples ({TRAIN_SUBSET_PERCENTAGE*100:.1f}% of train).")
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]
    dataset_subset_to_process = [full_train_dataset[i] for i in subset_indices]
    logging.info(f"Subset created. Processing {len(dataset_subset_to_process)} items.")

    # 3. Process Data in Parallel and Write to CSV
    approved_count = 0
    processed_count = 0
    if args.task == "single_gene_prediction" or args.task == "metabolic_flux" or args.task == "seahorse_cluster":

        output_fieldnames = [
            "pert", "gene", "cell_type",     # Original context
            "user_query",                    # Input prompt to generator
            "true_answer",                   # Ground truth label (for reference)
            "user_prompt",                   # Store user prompt
            "assistant_response",            # Full response from the Generator
            "generated_thinking",            # Extracted thinking part from Generator
            "generated_answer",              # Extracted answer part from Generator             # Full response from the Critic LLM
            "critic_evaluation",             # Parsed 'good'/'bad' evaluation from Critic
            "critic_reasoning"               # Parsed justification from Critic
        ]

    elif args.task == "direct_prediction":
        output_fieldnames = [
            "pert", "solution", "cell_type", "raw_solution_lists",     # Original context
            "user_query",                    # Input prompt to generator
            "true_answer",                   # Ground truth label (for reference)
            "user_prompt",                   # Store user prompt
            "assistant_response",            # Full response from the Generator
            "generated_thinking",            # Extracted thinking part from Generator
            "generated_answer",              # Extracted answer part from Generator             # Full response from the Critic LLM
            "critic_evaluation",             # Parsed 'good'/'bad' evaluation from Critic
            "critic_reasoning"               # Parsed justification from Critic
        ]
    elif args.task == "inverse_problem":
        output_fieldnames = [
            "cell_type",                    # Input context
            "input_upregulated_genes",      # Input context
            "input_downregulated_genes",    # Input context
            "true_pert",                    # Ground truth label
            "user_prompt",                  # The prompt sent to the generator
            "assistant_response",           # Full response from the Generator
            "generated_thinking",           # Extracted reasoning from Generator
            "generated_answer",             # The generator's predicted perturbation
            "critic_evaluation",            # Parsed 'good'/'bad' evaluation from Critic
            "critic_reasoning"              # Parsed justification from Critic
        ]
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

    try:
        with open(OUTPUT_CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output_fieldnames)
            writer.writeheader()
            logging.info(f"Opened {OUTPUT_CSV_FILE} for writing.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                        executor.submit(
                            process_data_item,
                            data,
                            generator_model_name=GENERATOR_MODEL_NAME,
                            critic_model_name=CRITIC_MODEL_NAME,
                            base_url=BASE_URL,
                            api_key="xxxxx",
                            task=args.task,
                            list_format=args.list_format,
                            critic_acceptance_threshold=args.critic_acceptance_threshold
                        ): data
                        for data in dataset_subset_to_process
                    }
                logging.info(f"Submitted {len(futures)} tasks to the thread pool.")

                progress_bar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Generating & Critiquing",
                    file=sys.stdout
                    )
                for future in progress_bar:
                    processed_count += 1

                    result = future.result()
                    if result is not None: # Check if critic approved (process_data_item returned data)
                        writer.writerow(result)
                        approved_count += 1
                        # Flush occasionally
                        if approved_count % 200 == 0: file.flush()

                    # Update progress bar
                    approval_rate = (approved_count / processed_count * 100) if processed_count > 0 else 0
                    progress_bar.set_postfix({
                        "Approved": approved_count,
                        "Processed": processed_count,
                        "Rate": f"{approval_rate:.2f}%"
                    })


            file.flush() # Final flush
            logging.info("Processing complete.")

    except IOError as e:
        logging.error(f"Error opening or writing to CSV file {OUTPUT_CSV_FILE}: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during parallel processing: {e}")
        exit(1)

    # 4. Final Summary
    logging.info("--- Synthetic Data Generation Summary (Critic Based) ---")
    logging.info(f"Processed {processed_count} / {len(dataset_subset_to_process)} items from the subset.")
    logging.info(f"Generated {approved_count} examples approved by the critic.")
    if processed_count > 0:
        final_approval_rate = (approved_count / processed_count) * 100
        logging.info(f"Critic Approval Rate (on subset): {final_approval_rate:.2f}%")
    else:
        logging.info("No items were processed successfully.")
    logging.info(f"Critic-approved examples saved to: {OUTPUT_CSV_FILE}")
    logging.info("--- Script Finished ---")