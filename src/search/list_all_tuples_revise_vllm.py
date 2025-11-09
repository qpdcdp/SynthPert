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
    """Formats the prompt for the revision task."""
    revision_header = (
        f"You are acting as an expert biologist reviewing predictions from a preliminary analysis of an experiment where the gene '{pert}' was knocked down in '{cell_type}' cells.\n\n"
        "Your goal is to improve the quality of the list by correcting any potential errors. The preliminary analysis is a good baseline, but likely contains inaccuracies and incorrect predictions.\n\n"
        "Carefully review each initial prediction. If you believe a prediction is incorrect or could be improved, provide a corrected entry in the XML format below. If you agree with the initial prediction, do not include it in your response.\n\n"
        "--- START OF INITIAL PREDICTIONS ---\n"
    )
    predictions_text_block = [f"- Gene: {item['gene']}\n- Prediction:\n{item['prediction']}" for item in initial_predictions]
    predictions_text_block_str = "\n\n".join(predictions_text_block)
    revision_footer = ("\n--- END OF INITIAL PREDICTIONS ---\n\n" "Now, provide your complete, revised list of predictions for ALL genes in the following XML format. " "Do not include any other text outside the main <revisions> block.\n" "Example format:\n" "<revisions>\n" "  <prediction>\n" "    <gene>GENE_A</gene>\n" "    <answer>downregulated</answer>\n" "  </prediction>\n" "  <prediction>\n" "    <gene>GENE_B</gene>\n" "    <answer>upregulated</answer>\n" "  </prediction>\n" "</revisions>")
    return revision_header + predictions_text_block_str + revision_footer

def grade_revision(initial_predictions, revision_xml_str):
    """Grades the accuracy before and after the revision step."""
    initial_answers = {}
    for pred in initial_predictions:
        gene = pred['gene']
        ground_truth = pred['ground_truth_label']
        match = re.search(r"<answer>(.*?)</answer>", pred['prediction'], re.IGNORECASE | re.DOTALL)
        initial_pred_answer = match.group(1).strip().lower() if match else "parsing_failed"
        initial_answers[gene] = {"prediction": initial_pred_answer, "ground_truth": ground_truth}
    
    revised_answers = {}
    try:
        xml_match = re.search(r"<revisions>.*?</revisions>", revision_xml_str, re.DOTALL)
        if xml_match:
            clean_xml_str = xml_match.group(0)
            root = ET.fromstring(clean_xml_str)
            for pred_node in root.findall('prediction'):
                gene_node, answer_node = pred_node.find('gene'), pred_node.find('answer')
                if gene_node is not None and answer_node is not None:
                    revised_answers[gene_node.text.strip()] = answer_node.text.strip().lower()
        else: raise ET.ParseError
    except ET.ParseError:
        return {"error": "Failed to parse revision XML.", "pre_revision_accuracy": -1, "post_revision_accuracy": -1}

    initial_correct, revised_correct, total_genes = 0, 0, len(initial_answers)
    if total_genes == 0: return {"pre_revision_accuracy": 100.0, "post_revision_accuracy": 100.0, "genes_evaluated": 0, "initial_correct": 0, "revised_correct": 0}
    
    for gene, data in initial_answers.items():
        if data["prediction"] == data["ground_truth"]: initial_correct += 1
        if gene in revised_answers and revised_answers[gene] == data["ground_truth"]: revised_correct += 1
        
    return {"pre_revision_accuracy": (initial_correct / total_genes) * 100, "post_revision_accuracy": (revised_correct / total_genes) * 100, "genes_evaluated": total_genes, "initial_correct": initial_correct, "revised_correct": revised_correct}

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

    total_revisions_made = 0
    total_genes_processed = 0
    
    ### CHANGE START ###
    # New list to store detailed information about each revision
    revisions_log = []
    ### CHANGE END ###

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
        revision_chunk_size = args.revision_chunk_size
        final_revised_answers_map = {}
        all_revision_responses_text = []

        for i in range(0, len(initial_predictions), revision_chunk_size):
            prediction_chunk_for_revision = initial_predictions[i:i+revision_chunk_size]
            
            if not prediction_chunk_for_revision:
                continue

            revision_prompt_chunk = format_revision_prompt(cell_type, pert, prediction_chunk_for_revision)
            formatted_revision_prompt = reviser_tokenizer.apply_chat_template(
                [{"role": "user", "content": revision_prompt_chunk}], tokenize=False, add_generation_prompt=True
            )
            estimated_tokens = 40 * len(prediction_chunk_for_revision) + 200
            max_revision_tokens = min(estimated_tokens, 8192)
            reviser_params = SamplingParams(temperature=args.revision_temperature, max_tokens=max_revision_tokens)
            revision_outputs = reviser_model.generate([formatted_revision_prompt], reviser_params, use_tqdm=False)
            revision_response_chunk = revision_outputs[0].outputs[0].text.strip()
            all_revision_responses_text.append(revision_response_chunk)
            
            try:
                xml_match = re.search(r"<revisions>.*?</revisions>", revision_response_chunk, re.DOTALL)
                if xml_match:
                    root = ET.fromstring(xml_match.group(0))
                    for pred_node in root.findall('prediction'):
                        gene_node, answer_node = pred_node.find('gene'), pred_node.find('answer')
                        if gene_node is not None and answer_node is not None and gene_node.text:
                            final_revised_answers_map[gene_node.text.strip()] = answer_node.text.strip().lower()
            except ET.ParseError:
                print(f"WARN: Failed to parse XML chunk for {cell_type}/{pert}. Skipping chunk.")

        # --- 3. GRADING (Adapted for chunked revision results) ---
        initial_correct, revised_correct, total_genes = 0, 0, len(initial_predictions)
        if total_genes > 0:
            changes_count = 0
            for pred_item in initial_predictions:
                gene, ground_truth = pred_item['gene'], pred_item['ground_truth_label']
                match = re.search(r"<answer>(.*?)</answer>", pred_item['prediction'], re.IGNORECASE | re.DOTALL)
                initial_pred_answer = match.group(1).strip().lower() if match else "parsing_failed"
                
                all_true_labels_initial.append(ground_truth)
                all_predicted_labels_initial.append(initial_pred_answer)
                if initial_pred_answer == ground_truth:
                    initial_correct += 1

                revised_pred_answer = final_revised_answers_map.get(gene, initial_pred_answer)

                all_true_labels_revised.append(ground_truth)
                all_predicted_labels_revised.append(revised_pred_answer)
                if revised_pred_answer == ground_truth:
                    revised_correct += 1
                
                if initial_pred_answer != revised_pred_answer:
                    changes_count += 1
                    ### CHANGE START ###
                    # Log the specific change that was made
                    revisions_log.append({
                        "cell_type": cell_type,
                        "pert": pert,
                        "gene": gene,
                        "initial_prediction": initial_pred_answer,
                        "revised_prediction": revised_pred_answer,
                        "ground_truth": ground_truth
                    })
                    ### CHANGE END ###
            
            total_revisions_made += changes_count
            total_genes_processed += total_genes

            print(f"[{cell_type}/{pert}] Revisions made: {changes_count} / {total_genes} predictions were changed.")
            
            grading_stats = {
                "pre_revision_accuracy": (initial_correct / total_genes) * 100 if total_genes > 0 else 100.0,
                "post_revision_accuracy": (revised_correct / total_genes) * 100 if total_genes > 0 else 100.0,
                "genes_evaluated": total_genes, 
                "initial_correct": initial_correct, 
                "revised_correct": revised_correct,
                "revisions_made": changes_count
            }
            # This is where you would append `grading_stats` to your `results` list if you were saving per-group JSONL results.
            # Example: results.append({"cell_type": cell_type, "pert": pert, **grading_stats})

    # --- FINAL FILE WRITING ---
    print(f"\nWriting {len(results)} results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res) + '\n')
    print(f"Processing complete. Results saved to {output_path}")

    ### CHANGE START ###
    # --- WRITE REVISIONS LOG TO A SEPARATE TXT FILE ---
    revisions_path = Path(args.output_dir) / "revisions_made.txt"
    print(f"\nWriting {len(revisions_log)} actual revisions to {revisions_path}...")
    try:
        with open(revisions_path, 'w', encoding='utf-8') as f_rev:
            # Write a header for the tab-separated file
            f_rev.write("cell_type\tpert\tgene\tinitial_prediction\trevised_prediction\tground_truth\n")
            # Write each logged revision to the file
            for revision in revisions_log:
                f_rev.write(
                    f"{revision['cell_type']}\t"
                    f"{revision['pert']}\t"
                    f"{revision['gene']}\t"
                    f"{revision['initial_prediction']}\t"
                    f"{revision['revised_prediction']}\t"
                    f"{revision['ground_truth']}\n"
                )
        print(f"Successfully saved revisions log to {revisions_path}")
    except IOError as e:
        print(f"Error writing revisions log to file: {e}")
    ### CHANGE END ###


    print("\n\n--- OVERALL CLASSIFICATION REPORTS ---")
    labels = ["upregulated", "downregulated", "not differentially expressed"]

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

    print("\n--- Initial Predictor Report ---")
    print(initial_report)
    print("\n--- Revised Predictor Report ---")
    print(revised_report)

    report_path = Path(args.output_dir) / "classification_report.txt"

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

            if total_genes_processed > 0:
                change_percentage = (total_revisions_made / total_genes_processed) * 100
                f_report.write(f"Total Revisions Made Across All Groups: {total_revisions_made}\n")
                f_report.write(f"Total Genes Processed: {total_genes_processed}\n")
                f_report.write(f"Overall Percentage of Predictions Changed: {change_percentage:.2f}%\n")
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
    parser.add_argument("--revision_chunk_size", type=int, default=50, help="Number of genes to include in each revision prompt.")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="Sampling temperature for the initial prediction.")
    parser.add_argument("--revision_temperature", type=float, default=1.0, help="Sampling temperature for the revision step.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum new tokens for generation.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Number of GPUs for tensor parallelism. Defaults to all available GPUs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--csv_data_directory", type=str, default="./data", help="Directory containing the CSV files for the dataset.")
    parser.add_argument("--merged_model_dir", type=str, default="./output/merged_models/merged_model_for_vllm", help="Directory to save output files.")
    parser.add_argument("--test_split_cell_lines", action="store_true", help="Use cell lines for the test split.")
    parser.add_argument("--devices", type=int, default=8, help="Number of devices (GPUs) to use for tensor parallelism. Defaults to 1.")
    args = parser.parse_args()
    main(args)