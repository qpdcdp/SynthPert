import argparse
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import xml.etree.ElementTree as ET
import random
from sklearn.metrics import classification_report

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams
# Import VLLM

from src.data import DiffExpressionDataset

def load_vllm_model(model_path, lora_checkpoint, tensor_parallel_size):
    """Loads a vLLM engine and its tokenizer."""
    print(f"Loading model from: {model_path} using vLLM.")
    
    # vLLM automatically uses available GPUs.
    # If tensor_parallel_size is not specified, it defaults to all available GPUs.
    num_gpus = torch.cuda.device_count()
    tp_size = tensor_parallel_size if tensor_parallel_size is not None else num_gpus
    if tp_size > num_gpus:
        raise ValueError(f"Requested tensor_parallel_size ({tp_size}) is greater than the number of available GPUs ({num_gpus}).")


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = xxxx
    # vLLM handles padding internally, so padding_side is not strictly necessary
    # for the generation process itself but can be good practice.
    tokenizer.padding_side = "left"

    # --- Model Loading ---
    # vLLM loads the model. If a LoRA checkpoint is provided, we first
    # merge it with the base model and save it to a new directory.
    model_path_for_vllm = model_path

    if lora_checkpoint:
        
        merged_model_dir = Path(args.merged_model_dir) / args.predictor_model_path
        merged_model_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading base model for LoRA merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Loading and merging LoRA adapter...")
        lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint)
        model = lora_model.merge_and_unload()
        print(f"Saving merged model to {merged_model_dir}...")
        model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        print("Model merging and saving complete.")

        model_path_for_vllm = str(merged_model_dir)

    # Initialize vLLM
    # The number of GPUs is used for tensor parallelism.
    # vLLM handles model distribution automatically.
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path_for_vllm,
        tensor_parallel_size=args.devices,
        trust_remote_code=True,
        seed=args.seed,
        disable_log_stats=True
    )
    
    print("vLLM engine and tokenizer loaded successfully.")
    return llm, tokenizer

def format_revision_prompt(cell_type, pert, initial_predictions):
    """Formats the prompt for the 'revise by exception' task."""
    system_prompt = """You are a senior biologist reviewing gene expression predictions.

    The predictions you will receive are from an automated preliminary analysis which is known to produce a high number of **false positives**. Your task is not to review each gene in isolation, but to perform a **global coherence check**.

    Your most critical task is to assess if the **entire set of predicted gene regulations represents a plausible biological story** that would result from the specific gene perturbation (knockdown) mentioned in the experiment context. An individual prediction, even if plausible on its own, should be rejected if it does not fit the larger, coherent narrative of the cell's response to the perturbation.

    Based on this global coherence check, apply stringent scrutiny. Re-classify AS MANY GENES AS POSSIBLE to a more conservative or accurate state (e.g., "not differentially expressed") if they weaken the overall biological narrative.

    **Output Rules:**
    1.  Review the entire list of predictions in the context of the specific perturbation.
    2.  If you **disagree** with a prediction because it is individually unlikely or does not fit the coherent biological story, provide a corrected entry in the XML format below.
    3.  If you **agree** with a prediction, **DO NOT** include it in your response.
    4.  Your response must ONLY contain the genes whose predictions you wish to change.
    5.  If the entire set of predictions forms a coherent story and you agree with all of them, provide an empty <revisions></revisions> block.

    Example format for corrections:
    <revisions>
    <prediction>
        <gene>GENE_A</gene>
        <answer>not differentially expressed</answer>
    </prediction>
    <prediction>
        <gene>GENE_C</gene>
        <answer>upregulated</answer>
    </prediction>
    </revisions>
    """

    predictions_text_block = [f"- Gene: {item['gene']}\n- Prediction:\n{item['prediction']}" for item in initial_predictions]
    predictions_text_block_str = "\n\n".join(predictions_text_block)
    user_query = f"""Here is the data for the current review.

    **Experiment Context:**
    - Perturbed Gene (Knockdown): {pert}
    - Cell Type: {cell_type}

    --- START OF INITIAL PREDICTIONS ---

    {predictions_text_block_str}

    --- END OF INITIAL PREDICTIONS ---

    Now, provide your list of corrections based on the instructions in your system prompt.
    """
    return system_prompt, user_query


def grade_revision_by_exception(initial_predictions, revision_xml_str):
    """
    Grades accuracy by taking initial predictions and applying a sparse set of XML corrections.
    """
    # Step 1: Parse the initial predictions and ground truth into a dictionary
    initial_answers = {}
    for pred in initial_predictions:
        gene = pred['gene']
        ground_truth = pred['ground_truth_label']
        # Extract the initial prediction text from the <answer> tag
        match = re.search(r"<answer>(.*?)</answer>", pred['prediction'], re.IGNORECASE | re.DOTALL)
        initial_pred_answer = "parsing_failed"
        if match:
            # Clean up the answer
            ans = match.group(1).strip().lower()
            if "not differentially expressed" in ans:
                initial_pred_answer = "not differentially expressed"
            elif "upregulated" in ans:
                initial_pred_answer = "upregulated"
            elif "downregulated" in ans:
                initial_pred_answer = "downregulated"
        
        initial_answers[gene] = {"prediction": initial_pred_answer, "ground_truth": ground_truth}

    # Step 2: Create the final revised list, starting with the initial predictions
    # This is a crucial step: the revised list starts as a copy of the initial one.
    revised_answers = {gene: data["prediction"] for gene, data in initial_answers.items()}

    # Step 3: Parse the XML containing ONLY the changes
    try:
        xml_match = re.search(r"<revisions>.*?</revisions>", revision_xml_str, re.DOTALL)
        if xml_match:
            clean_xml_str = xml_match.group(0)
            root = ET.fromstring(clean_xml_str)
            for pred_node in root.findall('prediction'):
                gene_node = pred_node.find('gene')
                answer_node = pred_node.find('answer')
                if gene_node is not None and answer_node is not None and gene_node.text:
                    gene_to_update = gene_node.text.strip()
                    new_answer = answer_node.text.strip().lower()
                    # Step 4: Update the revised list with the corrections
                    if gene_to_update in revised_answers:
                        revised_answers[gene_to_update] = new_answer
                    else:
                        print(f"WARN: Gene '{gene_to_update}' from revision XML not found in initial prediction list. Ignoring.")

    except ET.ParseError:
        print(f"WARN: Failed to parse revision XML. No revisions will be applied. XML content:\n{revision_xml_str}")
        # If parsing fails, the revised list remains identical to the initial one.

    # Step 5: Grade both the initial and the final revised lists
    initial_correct = 0
    revised_correct = 0
    total_genes = len(initial_answers)

    if total_genes == 0:
        return {"pre_revision_accuracy": 100.0, "post_revision_accuracy": 100.0, "genes_evaluated": 0, "initial_correct": 0, "revised_correct": 0}

    for gene, data in initial_answers.items():
        # Grade initial prediction
        if data["prediction"] == data["ground_truth"]:
            initial_correct += 1
        
        # Grade revised prediction
        if revised_answers.get(gene) == data["ground_truth"]:
            revised_correct += 1

    return {
        "pre_revision_accuracy": (initial_correct / total_genes) * 100,
        "post_revision_accuracy": (revised_correct / total_genes) * 100,
        "genes_evaluated": total_genes,
        "initial_correct": initial_correct,
        "revised_correct": revised_correct,
        "final_predictions": revised_answers # Also return the final merged list
    }


def main(args):
    print("Starting vLLM-based prediction and revision pipeline.")

    # Load Model(s)
    predictor_model, predictor_tokenizer = load_vllm_model(args.predictor_model_path, args.lora_checkpoint, args.tensor_parallel_size)
    
    if args.reviser_model_path != args.predictor_model_path:
        reviser_model, reviser_tokenizer = load_vllm_model(args.reviser_model_path, args.lora_checkpoint, args.tensor_parallel_size)
    else:
        reviser_model, reviser_tokenizer = predictor_model, predictor_tokenizer
    

    # Load and prepare dataset
    full_dataset = DiffExpressionDataset(csv_dir=args.data_dir, split='test', test_split_cell_lines=args.cell_line, generate_all_non_de_samples=args.generate_all_non_de_samples, generate_4x_non_de_samples=args.generate_4x_non_de_samples)
    total_samples = len(full_dataset)
    sample_size = int(total_samples * args.dataset_fraction)
    indices = random.sample(range(total_samples), sample_size)
    df = pd.DataFrame([full_dataset[i] for i in indices])
    grouped_data = list(df.groupby(['cell_type', 'pert']))
    
    if args.limit_groups > 0:
        grouped_data = grouped_data[:args.limit_groups]

    output_path = Path(args.output_dir) / "revision_experiment_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []

    #total_genes_to_process = sum(len(group) for _, group in grouped_data)
    all_true_labels_initial = []
    all_predicted_labels_initial = []
    all_true_labels_revised = []
    all_predicted_labels_revised = []
    
    # <<< CHANGE 1: INITIALIZE A LIST TO STORE REVISED ENTRIES >>>
    all_revised_entries = []


    group_pbar = tqdm(grouped_data, desc="Processing Groups", unit="group")

    for (cell_type, pert), group in group_pbar:
        # Update the description to show the current group being processed.
        group_pbar.set_description(f"Processing: {cell_type}/{pert}")

        # --- 1. INITIAL PREDICTION ---
        predictor_prompts = [row['prompt'] for row in group.to_dict('records')]
        genes_in_group = [row['gene'] for row in group.to_dict('records')]
        ground_truth_labels_in_group = [full_dataset.label_map[row['label']] for row in group.to_dict('records')]
        
        prediction_texts = []
        if predictor_prompts:
            formatted_prompts = [
                predictor_tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in predictor_prompts
            ]
            predictor_params = SamplingParams(temperature=args.generation_temperature, max_tokens=args.max_tokens)
            outputs = predictor_model.generate(formatted_prompts, predictor_params, use_tqdm=False)
            prediction_texts = [output.outputs[0].text.strip() for output in outputs]
        
        initial_predictions = [{'gene': g, 'ground_truth_label': l, 'prediction': p} for g, l, p in zip(genes_in_group, ground_truth_labels_in_group, prediction_texts)]

        # --- 2. REVISION STEP ---
        all_revision_responses_text = []
        
        # This map will store the final, merged predictions after all chunks are processed
        # It starts as a copy of the initial predictions
        initial_preds_map = {
            item['gene']: re.search(r"<answer>(.*?)</answer>", item['prediction'], re.IGNORECASE | re.DOTALL).group(1).strip().lower() if re.search(r"<answer>(.*?)</answer>", item['prediction'], re.IGNORECASE | re.DOTALL) else "parsing_failed"
            for item in initial_predictions
        }
        final_revised_answers_map = initial_preds_map.copy()

        revision_chunk_size = len(group)
        
        for i in range(0, len(initial_predictions), revision_chunk_size):
            prediction_chunk_for_revision = initial_predictions[i:i+revision_chunk_size]
            
            if not prediction_chunk_for_revision:
                continue

            # a. Format a manageable revision prompt using the new function
            system_prompt, user_query = format_revision_prompt(cell_type, pert, prediction_chunk_for_revision)

            # b. Format the single prompt string
            formatted_revision_prompt = reviser_tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}], tokenize=False, add_generation_prompt=True
            )

            # c. Calculate a safe max_tokens for the revision response.
            # This can be much smaller now, as we expect fewer responses.
            # Let's estimate 40 tokens per gene, but for at most half the genes + buffer.
            estimated_tokens = int(40 * len(prediction_chunk_for_revision) * 0.5) + 200
            max_revision_tokens = min(estimated_tokens, 4096) # Cap at a reasonable limit

            reviser_params = SamplingParams(temperature=args.revision_temperature, max_tokens=args.max_tokens)
            
            # d. Generate the revision
            revision_outputs = reviser_model.generate([formatted_revision_prompt], reviser_params, use_tqdm=False)
            revision_response_chunk = revision_outputs[0].outputs[0].text.strip()
            all_revision_responses_text.append(revision_response_chunk)
            
            # e. Parse the XML chunk and update the final answers map
            try:
                xml_match = re.search(r"<revisions>.*?</revisions>", revision_response_chunk, re.DOTALL)
                if xml_match:
                    root = ET.fromstring(xml_match.group(0))
                    for pred_node in root.findall('prediction'):
                        gene_node = pred_node.find('gene')
                        answer_node = pred_node.find('answer')
                        if gene_node is not None and answer_node is not None and gene_node.text:
                            gene_to_update = gene_node.text.strip()
                            new_answer = answer_node.text.strip().lower()
                            if gene_to_update in final_revised_answers_map:
                                final_revised_answers_map[gene_to_update] = new_answer
            except ET.ParseError:
                print(f"WARN: Failed to parse XML chunk for {cell_type}/{pert}. Skipping chunk.")

        
        # --- 3. GRADING AND TRACKING REVISIONS ---
        initial_correct, revised_correct, total_genes = 0, 0, len(initial_predictions)
        if total_genes > 0:
            for pred_item in initial_predictions:
                gene = pred_item['gene']
                ground_truth = pred_item['ground_truth_label']
                
                # Get initial prediction
                initial_pred_answer = initial_preds_map.get(gene, "parsing_failed")
                all_true_labels_initial.append(ground_truth)
                all_predicted_labels_initial.append(initial_pred_answer)
                if initial_pred_answer == ground_truth:
                    initial_correct += 1

                # Get revised prediction
                revised_pred_answer = final_revised_answers_map.get(gene, "parsing_failed")
                all_true_labels_revised.append(ground_truth)
                all_predicted_labels_revised.append(revised_pred_answer)
                if revised_pred_answer == ground_truth:
                    revised_correct += 1

                if initial_pred_answer != revised_pred_answer:
                    all_revised_entries.append({
                        "cell_type": cell_type,
                        "perturbation": pert,
                        "gene": gene,
                        "initial_prediction": initial_pred_answer,
                        "revised_prediction": revised_pred_answer,
                        "formatted_revision_prompt": formatted_revision_prompt  # <-- ADD THIS LINE
                    })

            grading_stats = {
                "pre_revision_accuracy": (initial_correct / total_genes) * 100,
                "post_revision_accuracy": (revised_correct / total_genes) * 100,
                "genes_evaluated": total_genes, "initial_correct": initial_correct, "revised_correct": revised_correct
            }
        else:
            grading_stats = {"pre_revision_accuracy": 100.0, "post_revision_accuracy": 100.0, "genes_evaluated": 0, "initial_correct": 0, "revised_correct": 0}

        # For logging, combine all partial revision responses
        full_revision_response_log = "\n\n--- REVISION CHUNK ---\n\n".join(all_revision_responses_text)

        # --- 4. SAVING RESULTS ---
        results.append({
            'cell_type': cell_type, 
            'perturbation': pert, 
            'grading_stats': grading_stats, 
            'initial_predictions': initial_predictions, 
            'revision_response': full_revision_response_log,
            'revision_prompt': formatted_revision_prompt
        })
    
    

    # --- FINAL FILE WRITING ---
    print(f"\nWriting {len(results)} results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res) + '\n')
    print(f"Processing complete. Results saved to {output_path}")

    # <<< CHANGE 3: SAVE THE LOG OF ACTUAL REVISIONS TO A TEXT FILE >>>
    revised_log_path = Path(args.output_dir) / "revised_predictions_log.txt"
    print(f"\nSaving a log of {len(all_revised_entries)} actual revisions to {revised_log_path}...")
    try:
        with open(revised_log_path, 'w', encoding='utf-8') as f_revised:
            if not all_revised_entries:
                f_revised.write("No predictions were revised across all processed groups.\n")
            else:
                # Sort entries for a clean, grouped report
                sorted_entries = sorted(all_revised_entries, key=lambda x: (x['cell_type'], x['perturbation']))
                last_group = None
                for entry in sorted_entries:
                    current_group = (entry['cell_type'], entry['perturbation'])
                    if current_group != last_group:
                        f_revised.write(f"\n--- Group: {entry['cell_type']} / {entry['perturbation']} ---\n")
                        last_group = current_group
                    
                    f_revised.write(f"Gene: {entry['gene']}\n")
                    f_revised.write(f'  - Revision Prompt: {entry["formatted_revision_prompt"]}')
                    f_revised.write(f"  - Initial Prediction: {entry['initial_prediction']}\n")
                    f_revised.write(f"  - Revised Prediction: {entry['revised_prediction']}\n")
        print("Revision log saved successfully.")
    except IOError as e:
        print(f"Error writing revision log to file: {e}")


    print("\n\n--- OVERALL CLASSIFICATION REPORTS ---")
    # Define the possible labels to ensure consistent report ordering
    labels = ["upregulated", "downregulated", "not differentially expressed", "parsing_failed"]

    # Generate classification reports as strings
    initial_report = classification_report(
        all_true_labels_initial,
        all_predicted_labels_initial,
        labels=labels,
        digits=4,
        zero_division=0
    )
    revised_report = classification_report(
        all_true_labels_revised,
        all_predicted_labels_revised,
        labels=labels,
        digits=4,
        zero_division=0
    )

    # Print reports to console
    print("\n--- Initial Predictor Report ---")
    print(initial_report)
    print("\n--- Revised Predictor Report ---")
    print(revised_report)

    # Determine the report file path
    report_path = Path(args.output_dir) / "classification_report_with_revisions.txt"

    # Write reports to the file
    print(f"\nWriting classification reports to {report_path}...")
    try:
        with open(report_path, 'w', encoding='utf-8') as f_report:
            f_report.write("--- OVERALL CLASSIFICATION REPORTS ---\n\n")
            
            f_report.write("--- Initial Predictor Report ---\n")
            f_report.write(initial_report)
            f_report.write("\n\n")
            
            f_report.write("--- Revised Predictor Report ---\n")
            f_report.write(revised_report)
            f_report.write("\n")
        print(f"Successfully saved classification reports to {report_path}")
    except IOError as e:
        print(f"Error writing classification report to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a two-stage prediction and revision pipeline for gene expression analysis using vLLM.")
    parser.add_argument("--predictor_model_path", type=str, required=True, help="Path to the initial predictor model.")
    parser.add_argument("--reviser_model_path", type=str, required=True, help="Path to the reviser model.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset CSVs.")
    parser.add_argument("--output_dir", type=str, default="", help="Path to the output folder.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA adapter checkpoint to be used by the model(s).")
    parser.add_argument("--generate_all_non_de_samples", action="store_true", help="Use a 100x larger set of 'not differentially expressed' genes.")
    parser.add_argument("--generate_4x_non_de_samples", action="store_true", help="Use a 4x larger set of 'not differentially expressed' genes.")
    parser.add_argument("--limit_groups", type=int, default=0, help="Limit the number of (cell_type, perturbation) groups to process. 0 for all.")
    parser.add_argument("--cell_line", type=str, default="none", help="Specify cell lines for the test split.")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of the dataset to use (0.0 to 1.0).")
    parser.add_argument("--revision_chunk_size", type=int, default=100, help="Number of genes to include in each revision prompt.")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Sampling temperature for the initial prediction.")
    parser.add_argument("--revision_temperature", type=float, default=1.0, help="Sampling temperature for the revision step.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum new tokens for generation.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Number of GPUs for tensor parallelism. Defaults to all available GPUs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--csv_data_directory", type=str, default="./data", help="Directory containing the CSV files for the dataset.")
    parser.add_argument("--merged_model_dir", type=str, default="./output/merged_models/merged_model_for_vllm_exception", help="Directory to save output files.")
    parser.add_argument("--test_split_cell_lines", action="store_true", help="Use cell lines for the test split.")
    parser.add_argument("--devices", type=int, default=8, help="Number of devices (GPUs) to use for tensor parallelism. Defaults to 1.")
    args = parser.parse_args()
    main(args)