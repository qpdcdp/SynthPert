from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path
import json
import re
import torch
import sys
import random
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data import DiffExpressionDataset, GeneRegulationListDataset

from accelerate import Accelerator
from accelerate.utils import gather_object

def calculate_set_performance_metrics(pred_set, true_set):
    # Ensure inputs are sets
    pred_set = set(pred_set)
    true_set = set(true_set)
    
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    num_pred = len(pred_set)
    num_true = len(true_set)
    
    if num_pred == 0 and num_true == 0:
        precision, recall, f1 = 1.0, 1.0, 1.0
    else:
        precision = tp / num_pred if num_pred > 0 else 0.0
        recall = tp / num_true if num_true > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {"tp": tp, "fp": fp, "fn": fn, "num_pred": num_pred, "num_true": num_true, "precision": precision, "recall": recall, "f1": f1}

def main(args):
    csv_data_directory = "./data"
    output_dir = Path(args.output_dir)
    output_filename = "evaluation_results.jsonl"

    accelerator = Accelerator()
    accelerator.print(f"Process {accelerator.process_index} of {accelerator.num_processes} using device: {accelerator.device}")
    
    # Create output directory on all processes to ensure it exists for saving temp files
    output_dir.mkdir(parents=True, exist_ok=True)

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.lora_checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = xxxx
        tokenizer.padding_side = "left"
        
        accelerator.print(f"Loading base model from: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )
    
        if args.lora_checkpoint:
            accelerator.print(f"Loading and merging LoRA adapter from: {args.lora_checkpoint}")
            model = PeftModel.from_pretrained(model, args.lora_checkpoint)
            model = model.merge_and_unload()
            accelerator.print("Adapter merged successfully.")
        model.eval()

    if accelerator.is_main_process:
        print("Model and Tokenizer loaded.")

    test_dataset = GeneRegulationListDataset(csv_dir=csv_data_directory, split="test", test_split_cell_lines=args.test_split_cell_lines)
    sample_size = int(len(test_dataset) * args.dataset_fraction)
    # Ensure random sampling is consistent across processes if needed, but for subsetting it's fine.
    indices = random.sample(range(len(test_dataset)), sample_size)
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    if accelerator.is_main_process:
         print(f"Test dataset size (after fraction sampling): {len(test_dataset)} samples.")

    def extract_answer(generated_text):
        if not isinstance(generated_text, str): return None
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
        if not match: return {'upregulated': [], 'downregulated': []} # Return empty dict if no answer tag
        
        answer = match.group(1).strip()
        result = {'upregulated': [], 'downregulated': []}
        up_match = re.search(r'Upregulated:\s*(.*?)(?=Downregulated:|$)', answer, re.IGNORECASE | re.DOTALL)
        down_match = re.search(r'Downregulated:\s*(.*)', answer, re.IGNORECASE | re.DOTALL)
        if up_match:
            up_content = up_match.group(1).strip()
            if up_content and up_content.upper() not in ["NONE", "N/A"]:
                result['upregulated'] = [gene.strip() for gene in re.split(r'[,;\n]', up_content) if gene.strip()]
        if down_match:
            down_content = down_match.group(1).strip()
            if down_content and down_content.upper() not in ["NONE", "N/A"]:
                result['downregulated'] = [gene.strip() for gene in re.split(r'[,;\n]', down_content) if gene.strip()]
        return result

    def collate_fn(batch):
        prompts_raw = [item['prompt'] for item in batch]
        prompts_for_tokenizer = []
        for p in prompts_raw:
            # Using a mock template function as the original is not provided
            prompts_for_tokenizer.append(f"User: {p[0]['content']}\nAssistant:")
        
        tokenized_inputs = tokenizer(prompts_for_tokenizer, padding=True, truncation=True, return_tensors="pt", max_length=4096)
        return {
            "input_ids": tokenized_inputs.input_ids, "attention_mask": tokenized_inputs.attention_mask,
            "solutions": [item.get('raw_solution_lists') for item in batch],
            "perts": [item.get('pert') for item in batch], "genes": [item.get('gene') for item in batch],
            "cell_types": [item.get('cell_type') for item in batch],
        }
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    results_local = []
    progress_bar = tqdm(test_dataloader, desc=f"Rank {accelerator.process_index} Eval", disable=not accelerator.is_local_main_process, file=sys.stdout)

    with torch.no_grad():
        for batch in progress_bar:
            outputs = accelerator.unwrap_model(model).generate(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                max_new_tokens=2048, pad_token_id=tokenizer.pad_token_id,
            )
            generated_ids = outputs[:, batch['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, gen_text in enumerate(generated_texts):
                extracted_answer = extract_answer(gen_text)
                correct_solution = batch['solutions'][i] if batch['solutions'][i] else {'upregulated': [], 'downregulated': []}
                
                up_metrics = calculate_set_performance_metrics(extracted_answer['upregulated'], correct_solution['upregulated'])
                down_metrics = calculate_set_performance_metrics(extracted_answer['downregulated'], correct_solution['downregulated'])
                
                results_local.append({
                    "pert": batch['perts'][i], "cell_type": batch['cell_types'][i],
                    "generated_text": gen_text, "extracted_answer": extracted_answer,
                    "correct_solution": correct_solution,
                    "is_correct_strict": up_metrics["f1"] == 1.0 and down_metrics["f1"] == 1.0,
                    "upregulated_metrics": up_metrics, "downregulated_metrics": down_metrics
                })

    # ========= FIX HIGHLIGHT: SAVE RESULTS LOCALLY =========
    # Each process saves its own results to a temporary file. This avoids the
    # massive communication overhead and potential timeout of `gather_object`.
    temp_results_file = output_dir / f"temp_results_rank_{accelerator.process_index}.jsonl"
    with open(temp_results_file, 'w', encoding='utf-8') as f:
        for item in results_local:
            f.write(json.dumps(item) + '\n')
            
    accelerator.print(f"Rank {accelerator.process_index} finished eval and saved {len(results_local)} results to {temp_results_file}")

    # ========= FIX HIGHLIGHT: SYNCHRONIZE ALL PROCESSES =========
    # Wait for all processes to finish writing their temporary files before proceeding.
    accelerator.wait_for_everyone()

    # ========= FIX HIGHLIGHT: AGGREGATE ON MAIN PROCESS =========
    if accelerator.is_main_process:
        print(f"\n--- Post-processing on Main Process ---")
        all_results = []
        # Loop through all the temporary files created by each process
        for i in range(accelerator.num_processes):
            temp_file_path = output_dir / f"temp_results_rank_{i}.jsonl"
            try:
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line))
                os.remove(temp_file_path) # Clean up the temporary file
            except FileNotFoundError:
                print(f"Warning: Could not find temporary file {temp_file_path}. That process may have failed.")

        print(f"Gathered and assembled {len(all_results)} total results from {accelerator.num_processes} processes.")

        if not all_results:
            print("No results were generated. Exiting.")
            return

        total_correct = sum(r['is_correct_strict'] for r in all_results)
        total_evaluated = len(all_results)
        
        print("\n--- Evaluation Summary ---")
        print(f"Model: {args.model_name_or_path}")
        print(f"Total Samples Evaluated: {total_evaluated}")
        print(f"Strict Accuracy (Both lists perfect): {(total_correct / total_evaluated * 100):.2f}%")
        
        if args.task == "direct_prediction":
            # --- Upregulated Genes ---
            total_tp_up = sum(r["upregulated_metrics"]["tp"] for r in all_results)
            total_num_pred_up = sum(r["upregulated_metrics"]["num_pred"] for r in all_results)
            total_num_true_up = sum(r["upregulated_metrics"]["num_true"] for r in all_results)
            f1_scores_up = [r["upregulated_metrics"]["f1"] for r in all_results]

            micro_p_up = total_tp_up / total_num_pred_up if total_num_pred_up > 0 else 0
            micro_r_up = total_tp_up / total_num_true_up if total_num_true_up > 0 else 0
            micro_f1_up = 2 * (micro_p_up * micro_r_up) / (micro_p_up + micro_r_up) if (micro_p_up + micro_r_up) > 0 else 0
            macro_f1_up = sum(f1_scores_up) / total_evaluated if total_evaluated > 0 else 0.0

            print(f"\n--- Upregulated Set Performance ---")
            print(f"  Overall Precision (Micro): {micro_p_up*100:.2f}%")
            print(f"  Overall Recall (Micro):    {micro_r_up*100:.2f}%")
            print(f"  Overall F1-Score (Micro):  {micro_f1_up:.4f}")
            print(f"  Average F1-Score (Macro):  {macro_f1_up:.4f}")

            # --- Downregulated Genes ---
            total_tp_down = sum(r["downregulated_metrics"]["tp"] for r in all_results)
            total_num_pred_down = sum(r["downregulated_metrics"]["num_pred"] for r in all_results)
            total_num_true_down = sum(r["downregulated_metrics"]["num_true"] for r in all_results)
            f1_scores_down = [r["downregulated_metrics"]["f1"] for r in all_results]

            micro_p_down = total_tp_down / total_num_pred_down if total_num_pred_down > 0 else 0
            micro_r_down = total_tp_down / total_num_true_down if total_num_true_down > 0 else 0
            micro_f1_down = 2 * (micro_p_down * micro_r_down) / (micro_p_down + micro_r_down) if (micro_p_down + micro_r_down) > 0 else 0
            macro_f1_down = sum(f1_scores_down) / total_evaluated if total_evaluated > 0 else 0.0

            print(f"\n--- Downregulated Set Performance ---")
            print(f"  Overall Precision (Micro): {micro_p_down*100:.2f}%")
            print(f"  Overall Recall (Micro):    {micro_r_down*100:.2f}%")
            print(f"  Overall F1-Score (Micro):  {micro_f1_down:.4f}")
            print(f"  Average F1-Score (Macro):  {macro_f1_down:.4f}")

        # --- Save Results to File ---
        output_path = output_dir / output_filename
        print(f"\nSaving {len(all_results)} detailed results to {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result_item in all_results:
                    f.write(json.dumps(result_item) + '\n')
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to file: {e}")

    # Final barrier to ensure main process finishes before other processes exit
    accelerator.wait_for_everyone()
    accelerator.print(f"Rank {accelerator.process_index} finished.")
    
    
if __name__ == "__main__":
    print("Starting evaluation script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dataset_fraction", type=float, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--list_format", type=str, required=True)
    parser.add_argument("--test_split_cell_lines", type=str, required=True)
    # Dummy args to accept them from the shell script
    parser.add_argument("--AUROC", action="store_true")
    parser.add_argument("--AUROC_stage", type=str, default=None)
    parser.add_argument("--test_script", type=str, default=None)
    parser.add_argument("--tool", type=str, default=None)
    args = parser.parse_args()
    main(args)