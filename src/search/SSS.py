import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

import pandas as pd
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import warnings
import json
import numpy as np


import random
from collections import defaultdict
import math
import os # For potential environment variable checking
import re

from src.data import DiffExpressionDataset, GeneRegulationListDataset


# --- Placeholder Prompt Loading Functions ---
# Replace these with your actual prompt loading mechanism (e.g., reading from files)
# These templates MUST contain the placeholders used in the format() calls later
# ({question}, {candidate}, {history}, {s1}, {s2})

# --- F1 Score Calculation Functions ---
def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score manually without using sklearn.
    
    Args:
        y_true: numpy array of true labels (0s and 1s)
        y_pred: numpy array of predicted labels (0s and 1s)
    
    Returns:
        f1_score: float value between 0 and 1
    """
    # Calculate True Positives, False Positives, and False Negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def calculate_gene_lists_f1(ground_truth, predicted):
    """
    Calculate F1 scores for gene regulation prediction task.
    
    Args:
        ground_truth: dict with 'upregulated' and 'downregulated' keys containing gene lists
        predicted: dict with 'upregulated' and 'downregulated' keys containing gene lists
    
    Returns:
        dict with 'upregulated_f1', 'downregulated_f1', and 'average_f1' scores
    """
    if ground_truth is None or predicted is None:
        return {"upregulated_f1": 0.0, "downregulated_f1": 0.0, "average_f1": 0.0}
    
    # Ensure both inputs have the required keys
    if not isinstance(ground_truth, dict) or not isinstance(predicted, dict):
        return {"upregulated_f1": 0.0, "downregulated_f1": 0.0, "average_f1": 0.0}
    
    # Get gene lists, defaulting to empty lists if keys are missing
    gt_up = ground_truth.get('upregulated', [])
    gt_down = ground_truth.get('downregulated', [])
    pred_up = predicted.get('upregulated', [])
    pred_down = predicted.get('downregulated', [])
    
    # Convert to sets for easier processing
    gt_up_set = set(gt_up) if gt_up else set()
    gt_down_set = set(gt_down) if gt_down else set()
    pred_up_set = set(pred_up) if pred_up else set()
    pred_down_set = set(pred_down) if pred_down else set()
    
    # Create a universe of all genes
    all_genes = gt_up_set | gt_down_set | pred_up_set | pred_down_set
    all_genes_list = list(all_genes)
    
    if not all_genes_list:
        # If no genes at all, perfect score
        return {"upregulated_f1": 1.0, "downregulated_f1": 1.0, "average_f1": 1.0}
    
    # For upregulated genes
    y_true_up = np.array([1 if gene in gt_up_set else 0 for gene in all_genes_list])
    y_pred_up = np.array([1 if gene in pred_up_set else 0 for gene in all_genes_list])
    
    # For downregulated genes
    y_true_down = np.array([1 if gene in gt_down_set else 0 for gene in all_genes_list])
    y_pred_down = np.array([1 if gene in pred_down_set else 0 for gene in all_genes_list])
    
    # Handle edge cases (empty lists) explicitly
    # For upregulated genes
    if np.sum(y_true_up) == 0 and np.sum(y_pred_up) == 0:
        up_f1 = 1.0  # Perfect agreement on empty lists
    elif np.sum(y_true_up) == 0 or np.sum(y_pred_up) == 0:
        up_f1 = 0.0  # One list is empty, the other isn't
    else:
        up_f1 = calculate_f1_score(y_true_up, y_pred_up)
    
    # For downregulated genes
    if np.sum(y_true_down) == 0 and np.sum(y_pred_down) == 0:
        down_f1 = 1.0  # Perfect agreement on empty lists
    elif np.sum(y_true_down) == 0 or np.sum(y_pred_down) == 0:
        down_f1 = 0.0  # One list is empty, the other isn't
    else:
        down_f1 = calculate_f1_score(y_true_down, y_pred_down)
    
    # Calculate average F1 score
    average_f1 = (up_f1 + down_f1) / 2
    
    return {
        "upregulated_f1": up_f1,
        "downregulated_f1": down_f1,
        "average_f1": average_f1
    }

verification_prompts_dirct_prediction = [
    # Turn 1: Initial analysis and biological reasoning framework
    "Verification Turn 1: Question:\n{question}\n\nCandidate Solution:\n{candidate}\n\n---\n"
    "Analyze the candidate solution's approach to identifying upregulated and downregulated genes. "
    "Deconstruct the reasoning. Then, rigorously outline how each of the following biological lemmas "
    "should be addressed with specific evidence and reasoning to determine the impact of the CRISPRi knockdown "
    "of gene {pert} in {cell_type} cells on other genes. Be exhaustive in your outline for each lemma:\n"
    "1. Gene Function Lemma - Known molecular function of {pert} product.\n"
    "2. Pathway Membership Lemma - Molecular pathways involving {pert} and potentially affected genes.\n"
    "3. Gene Perturbation Phenotypic Outcome Lemma - Known cellular phenotypic outcomes of modulating {pert}.\n"
    "4. Pathway Positioning Lemma - Functional role of {pert} (activator, mediator, repressor) in relevant pathways.\n"
    "5. Loci Regulatory Landscape Lemma - Regulatory landscape of potential target gene loci and TFs controlling them.\n"
    "6. Direct Regulation Lemma - Evidence of direct gene-expression regulatory connection between {pert} and other genes.\n"
    "7. Cell Type Context Lemma - Influence of {cell_type} on regulatory relationships.\n"
    "8. Analog Perturbations Modalities Lemma - Cellular responses to compounds/APIs targeting {pert} or its pathway.\n"
    "9. Temporal Dynamics Lemma - Expected timeframes for observing transcriptional effects after CRISPRi of {pert}.\n"
    "10. Secondary Effects Lemma - Potential indirect effects via intermediary genes or feedback loops from perturbing {pert}.\n"
    "11. Conflicting Evidence Lemma - Contradictory evidence in literature regarding {pert}'s interactions and effects.",

    # Turn 2: Synthesizing the gene regulation prediction
    "Verification Turn 2: History:\n{history}\n\n---\n"
    "Based on the established framework of biological lemmas from the previous turn, synthesize the "
    "argument for why specific genes are predicted to be upregulated or downregulated following CRISPRi of {pert}. "
    "Explicitly reference how each relevant biological lemma (by its number/name) supports the inclusion "
    "and predicted regulation status (upregulated/downregulated) of each gene. "
    "Conclude with a predicted list in the specified format: "
    "<answer>Upregulated: [list]; Downregulated: [list]</answer>.",

    # Turn 3: Detailed evidence for each biological lemma
    "Verification Turn 3: History:\n{history}\n\n---\n"
    "Provide detailed, step-by-step evidence and reasoning for each individual biological lemma "
    "as it pertains to the CRISPRi knockdown of {pert} in {cell_type} cells and its effect on potential target genes discussed. "
    "Ensure rigor, cite evidence where possible (conceptually, if actual DB lookups aren't done), and address all assumptions for each of the 11 lemmas.",

    # Turn 4: Error checking in the synthesized gene regulation prediction
    "Verification Turn 4: History:\n{history}\n\n---\n"
    "Scrutinize the synthesized gene regulation prediction (the list of up/downregulated genes and the reasoning) made in Turn 2. "
    "Verify each step in the reasoning, check if the biological lemmas were correctly applied and referenced, "
    "and identify any logical gaps, misinterpretation of biological principles, or unsubstantiated claims regarding specific gene regulation.",

    # Turn 5: Error checking in the evidence for biological lemmas
    "Verification Turn 5: History:\n{history}\n\n---\n"
    "Scrutinize the detailed evidence and reasoning provided for each biological lemma in Turn 3. "
    "Verify the biological accuracy of the statements, the logical soundness of the interpretations, "
    "and the correct application of biological knowledge for each of the 11 lemmas.",

    # Turn 6: Identify fatal errors in biological reasoning or prediction
    "Verification Turn 6: History:\n{history}\n\n---\n"
    "Based on your analysis in the previous turns, explicitly state if you found any 'fatal errors' in the biological reasoning or "
    "the predicted list of upregulated/downregulated genes. A fatal error is one that invalidates the predicted regulation status "
    "of one or more genes, or the core biological logic leading to the prediction. "
    "List any fatal errors found (e.g., 'Incorrect application of Pathway Positioning Lemma leading to wrong prediction for GENE_X', "
    "'Cited evidence for Direct Regulation Lemma does not support the claim for GENE_Y').",

    # Turn 7: Final Judgment on the candidate solution's gene list
    "Verification Turn 7: History:\n{history}\n\n---\n"
    "Final Judgment: Based *only* on the detailed biological analysis performed in the previous turns, "
    "rate the biological accuracy of the candidate solution's upregulated and downregulated gene lists "
    "on a scale of 0-100, where 100 means perfectly accurate and 0 means completely inaccurate. "
    "Consider both the inclusion of correct genes and the exclusion of incorrect genes. "
    "Respond in the exact format: 'Upregulated: X, Downregulated: Y' where X and Y are integer percentages."
    # # Turn 7: Final Judgment on the candidate solution's gene list
    # "Verification Turn 7: History:\n{history}\n\n---\n"
    # "Final Judgment: Based on your analysis, rate the biological accuracy of the upregulated and downregulated gene lists on a scale of 0-100."
    # "Format: 'Upregulated: X, Downregulated: Y' where X and Y are percentages."
]

verification_prompts_single_gene_prediction = [
    # Turn 1: Initial analysis and biological reasoning framework for the target gene
    "Verification Turn 1: Question:\n{question}\n\nCandidate Solution:\n{candidate}\n\n---\n"
    "Analyze the candidate solution's approach to predicting the regulatory effect on the target gene **{gene}** due to CRISPRi knockdown of **{pert}** in **{cell_type}** cells. "
    "Deconstruct the reasoning provided in the `<think>` tags. Then, rigorously outline how each of the following biological lemmas "
    "should be addressed with specific evidence and reasoning to determine the impact on the target gene **{gene}**. Be exhaustive in your outline for each lemma:\n"
    "1. Gene Function Lemma - Known molecular function of **{pert}** product.\n"
    "2. Pathway Membership Lemma - Molecular pathways involving **{pert}** and the target gene **{gene}**.\n"
    "3. Gene Perturbation Phenotypic Outcome Lemma - Known cellular phenotypic outcomes of modulating **{pert}** that might affect **{gene}**.\n"
    "4. Pathway Positioning Lemma - Functional role of **{pert}** (activator, mediator, repressor) in pathways relevant to **{gene}**.\n"
    "5. Loci Regulatory Landscape Lemma - Regulatory landscape of the genetic locus for **{gene}**, and TFs controlling its expression, potentially affected by **{pert}** knockdown.\n"
    "6. Direct Regulation Lemma - Evidence of direct gene-expression regulatory connection between **{pert}** and **{gene}**.\n"
    "7. Cell Type Context Lemma - Influence of **{cell_type}** on the regulatory relationship between **{pert}** and **{gene}**.\n"
    "8. Analog Perturbations Modalities Lemma - Cellular responses to compounds/APIs targeting **{pert}** or its pathway, and their known effects on genes like **{gene}**.\n"
    "9. Temporal Dynamics Lemma - Expected timeframes for observing transcriptional effects on **{gene}** after CRISPRi of **{pert}**.\n"
    "10. Secondary Effects Lemma - Potential indirect effects on **{gene}** via intermediary genes or feedback loops from perturbing **{pert}**.\n"
    "11. Conflicting Evidence Lemma - Contradictory evidence in literature regarding **{pert}**'s interaction with or influence on **{gene}**.",

    # Turn 2: Synthesizing the prediction for the single target gene
    "Verification Turn 2: History:\n{history}\n\n---\n"
    "Based on the established framework of biological lemmas from the previous turn, synthesize the "
    "argument for why the target gene **{gene}** is predicted to be 'upregulated', 'downregulated', or 'not differentially expressed' following CRISPRi of **{pert}**. "
    "Explicitly reference how each relevant biological lemma (by its number/name) supports this specific prediction for **{gene}**. "
    "Conclude with your own assessment in the specified format: "
    "<think>[Your synthesized reasoning here]</think><answer>[upregulated/downregulated/not differentially expressed]</answer>.",

    # Turn 3: Detailed evidence for each biological lemma regarding the target gene
    "Verification Turn 3: History:\n{history}\n\n---\n"
    "Provide detailed, step-by-step evidence and reasoning for each individual biological lemma "
    "as it pertains to the CRISPRi knockdown of **{pert}** in **{cell_type}** cells and its specific effect on the target gene **{gene}**. "
    "Ensure rigor, cite evidence where possible (conceptually), and address all assumptions for each of the 11 lemmas in the context of **{pert}** affecting **{gene}**.",

    # Turn 4: Error checking in the synthesized prediction for the target gene
    "Verification Turn 4: History:\n{history}\n\n---\n"
    "Scrutinize the synthesized prediction for the target gene **{gene}** (the reasoning in `<think>` tags and the final answer 'upregulated', 'downregulated', or 'not differentially expressed') made in Turn 2. "
    "Verify each step in the reasoning, check if the biological lemmas were correctly applied and referenced for **{gene}**, "
    "and identify any logical gaps, misinterpretation of biological principles, or unsubstantiated claims regarding the regulation of **{gene}**.",

    # Turn 5: Error checking in the evidence for biological lemmas
    "Verification Turn 5: History:\n{history}\n\n---\n"
    "Scrutinize the detailed evidence and reasoning provided for each biological lemma in Turn 3, specifically focusing on their application to the **{pert}** -> **{gene}** interaction. "
    "Verify the biological accuracy of the statements, the logical soundness of the interpretations, "
    "and the correct application of biological knowledge for each of the 11 lemmas concerning **{gene}**.",

    # Turn 6: Identify fatal errors in reasoning or prediction for the target gene
    "Verification Turn 6: History:\n{history}\n\n---\n"
    "Based on your analysis in the previous turns, explicitly state if you found any 'fatal errors' in the biological reasoning or "
    "the predicted regulatory status of the target gene **{gene}**. A fatal error is one that invalidates the predicted status for **{gene}** "
    "(e.g., claiming 'upregulated' when evidence points to 'downregulated' or 'not differentially expressed'), or if the core biological logic leading to the prediction for **{gene}** is flawed. "
    "List any fatal errors found (e.g., 'Incorrect application of Pathway Positioning Lemma leading to wrong prediction for **{gene}**', "
    "'Cited evidence for Direct Regulation Lemma does not support the claimed effect on **{gene}**').",

    #Turn 7: Final Judgment on the candidate solution's prediction for the target gene
    "Verification Turn 7: History:\n{history}\n\n---\n"
    "Final Judgment: Based *only* on the detailed biological analysis performed in the previous turns, "
    "is the original candidate solution's *final predicted status for the target gene **{gene}** (i.e., 'upregulated', 'downregulated', or 'not differentially expressed' within the `<answer>` tag) AND the reasoning within the `<think>` tag* "
    "ultimately correct and well-supported? "
    "Respond ONLY with the word 'Correct' or 'Incorrect'."
]

# verification_prompts = [  
#     # Turn 1: Initial analysis and rewrite request
#     "Verification Turn 1: Question:\n{question}\n\nCandidate Solution:\n{candidate}\n\n---\n"
#     "Analyze the candidate solution's approach. Rewrite it rigorously, breaking it down into "
#     "self-contained lemmas and a main theorem/proof structure. Be exhaustive.",
#     # Turn 2: Proof of main theorem using lemmas
#     "Verification Turn 2: History:\n{history}\n\n---\n"
#     "Now, write the main proof for the final answer, explicitly referencing the lemmas identified "
#     "or defined in the previous step. Ensure each step logically follows.",
#     # Turn 3: Proof of individual lemmas
#     "Verification Turn 3: History:\n{history}\n\n---\n"
#     "Provide detailed, step-by-step proofs for each individual lemma identified or defined previously. "
#     "Ensure rigor and address all assumptions.",
#     # Turn 4: Error checking in main proof
#     "Verification Turn 4: History:\n{history}\n\n---\n"
#     "Scrutinize the main proof written in Turn 2. Verify each step, check references to lemmas, "
#     "and identify any logical gaps, calculation errors, or misapplications of the lemmas.",
#     # Turn 5: Error checking in lemma proofs
#     "Verification Turn 5: History:\n{history}\n\n---\n"
#     "Scrutinize the lemma proofs written in Turn 3. Verify each step for correctness, logical soundness, "
#     "and calculation accuracy.",
#     # Turn 6: Identify fatal errors
#     "Verification Turn 6: History:\n{history}\n\n---\n"
#     "Based on your analysis in the previous turns, explicitly state if you found any 'fatal errors'. "
#     "A fatal error is one that invalidates the final answer or the core logic leading to it. "
#     "List any fatal errors found.",
#     # Turn 7: Final Judgment
#     "Verification Turn 7: History:\n{history}\n\n---\n"
#     "Final Judgment: Based *only* on the detailed analysis performed in the previous turns, "
#     "is the original candidate solution's *final answer* ultimately correct? "
#     "Respond ONLY with the word 'Correct' or 'Incorrect'."
# ]


comparison_prompts = [
    # Turn 1: Analysis and comparison
    "Comparison Turn 1: Question:\n{question}\n\n---\nSolution 1:\n{s1}\n\n---\nSolution 2:\n{s2}\n\n---\n"
    "Carefully analyze both solutions step-by-step. Identify key differences in approach, reasoning, "
    "calculations, and final answers. Discuss the potential correctness or flaws of each solution based on your analysis.",
    # Turn 2: Final Judgment
    "Comparison Turn 2: History:\n{history}\n\n---\n"
    "Based on your detailed analysis, which solution (1 or 2) is more likely correct or better reasoned? "
    "Respond ONLY with the number '1' or '2'."
]
# --- Utils ---

def extract_answer_from_response(generated_text):
    # Keep this function as is, assuming the API model follows the <answer> tag format
    if not isinstance(generated_text, str): # Add safety check
        return None
    match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Refined check for robustness
        if args.task == "single_gene_prediction":    
            if "upregulated" in answer: return "upregulated"
            if "downregulated" in answer: return "downregulated"
            # Handle "not differentially expressed" and "no"
            if "not differentially expressed" in answer: return "not differentially expressed"
            if "use gene enrichment libraries" in answer: return "use gene enrichment libraries"
        elif args.task == "direct_prediction" and args.list_format == "default":
            
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
        elif args.task == "direct_prediction" and args.list_format == "bullet_list":
            result = {'upregulated': [], 'downregulated': []}
            
            # Debug print for troubleshooting
            print(f"DEBUG: Processing answer text: {repr(answer[:200])}")
            
            # Handle the specific format: "Upregulated: gene1, gene2, ... Downregulated: gene3, gene4, ..."
            # First try to split by "Downregulated:" to separate the sections
            if "Downregulated:" in answer:
                parts = answer.split("Downregulated:", 1)
                up_section = parts[0]
                down_section = parts[1] if len(parts) > 1 else ""
                
                # Extract upregulated genes
                up_match = re.search(r'Upregulated:\s*(.*)', up_section, re.IGNORECASE | re.DOTALL)
                if up_match:
                    up_content = up_match.group(1).strip()
                    print(f"DEBUG: Up content: {repr(up_content)}")
                    
                    if up_content and up_content.upper() not in ["NONE", "N/A", "EMPTY", ""]:
                        # Handle comma-separated format
                        if ',' in up_content:
                            up_genes = [gene.strip() for gene in up_content.split(',') if gene.strip()]
                            result['upregulated'] = up_genes
                            print(f"DEBUG: Extracted upregulated genes: {up_genes}")
                
                # Extract downregulated genes
                down_content = down_section.strip()
                # Remove any trailing content after genes (like disclaimers)
                down_content = re.sub(r'\s*(Disclaimer|Note):.*$', '', down_content, flags=re.IGNORECASE | re.DOTALL)
                print(f"DEBUG: Down content: {repr(down_content)}")
                
                if down_content and down_content.upper() not in ["NONE", "N/A", "EMPTY", ""]:
                    # Handle comma-separated format
                    if ',' in down_content:
                        down_genes = [gene.strip() for gene in down_content.split(',') if gene.strip()]
                        result['downregulated'] = down_genes
                        print(f"DEBUG: Extracted downregulated genes: {down_genes}")
            
            # Fallback: try the original pattern-based approach
            if not result['upregulated'] and not result['downregulated']:
                print("DEBUG: Fallback to pattern-based extraction")
                
                # Multiple possible patterns for upregulated genes section
                up_patterns = [
                    r'UPREGULATED_GENES:\s*(.*?)(?=DOWNREGULATED_GENES:|Downregulated|$)',
                    r'UPREGULATED:\s*(.*?)(?=DOWNREGULATED:|Downregulated|$)', 
                    r'Upregulated\s*[Gg]enes?:?\s*(.*?)(?=Downregulated|$)',
                    r'Up-?regulated\s*[Gg]enes?:?\s*(.*?)(?=Down-?regulated|$)',
                ]
                
                # Multiple possible patterns for downregulated genes section
                down_patterns = [
                    r'DOWNREGULATED_GENES:\s*(.*?)(?=Disclaimer|$)',
                    r'DOWNREGULATED:\s*(.*?)(?=Disclaimer|$)',
                    r'Downregulated\s*[Gg]enes?:?\s*(.*?)(?=Disclaimer|$)',
                    r'Down-?regulated\s*[Gg]enes?:?\s*(.*?)(?=Disclaimer|$)',
                ]
                
                # Try to find upregulated genes section
                up_match = None
                for pattern in up_patterns:
                    up_match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
                    if up_match:
                        print(f"DEBUG: Up pattern matched: {pattern}")
                        break
                
                # Try to find downregulated genes section  
                down_match = None
                for pattern in down_patterns:
                    down_match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
                    if down_match:
                        print(f"DEBUG: Down pattern matched: {pattern}")
                        break
                
                # Process upregulated genes if found
                if up_match:
                    up_content = up_match.group(1).strip()
                    if up_content and up_content.upper() not in ["NONE", "N/A", "EMPTY", ""]:
                        up_genes = []
                        
                        # First try comma-separated format (most common)
                        if ',' in up_content:
                            # Split by commas and clean up
                            up_genes = [gene.strip() for gene in up_content.split(',') if gene.strip()]
                        else:
                            # Try different bullet point patterns
                            bullet_patterns = [
                                r'[-*•]\s*([^\n\r]+)',  # - gene, * gene, • gene
                                r'\d+\.\s*([^\n\r]+)',  # 1. gene, 2. gene
                                r'^\s*([A-Z][A-Z0-9_]+)\s*$',  # Just gene names on separate lines
                            ]
                            
                            for bullet_pattern in bullet_patterns:
                                matches = re.findall(bullet_pattern, up_content, re.MULTILINE)
                                if matches:
                                    up_genes.extend([gene.strip() for gene in matches if gene.strip()])
                                    break
                            
                            # If no bullet patterns worked, try splitting by lines and filtering
                            if not up_genes:
                                lines = [line.strip() for line in up_content.split('\n') if line.strip()]
                                # Filter for likely gene names (all caps, contains letters/numbers)
                                up_genes = [line for line in lines 
                                        if re.match(r'^[A-Z][A-Z0-9_-]*$', line) and len(line) > 1]
                        
                        result['upregulated'] = up_genes
                
                # Process downregulated genes if found
                if down_match:
                    down_content = down_match.group(1).strip()
                    if down_content and down_content.upper() not in ["NONE", "N/A", "EMPTY", ""]:
                        down_genes = []
                        
                        # First try comma-separated format (most common)
                        if ',' in down_content:
                            # Split by commas and clean up
                            down_genes = [gene.strip() for gene in down_content.split(',') if gene.strip()]
                        else:
                            # Try different bullet point patterns
                            bullet_patterns = [
                                r'[-*•]\s*([^\n\r]+)',  # - gene, * gene, • gene
                                r'\d+\.\s*([^\n\r]+)',  # 1. gene, 2. gene
                                r'^\s*([A-Z][A-Z0-9_]+)\s*$',  # Just gene names on separate lines
                            ]
                            
                            for bullet_pattern in bullet_patterns:
                                matches = re.findall(bullet_pattern, down_content, re.MULTILINE)
                                if matches:
                                    down_genes.extend([gene.strip() for gene in matches if gene.strip()])
                                    break
                            
                            # If no bullet patterns worked, try splitting by lines and filtering
                            if not down_genes:
                                lines = [line.strip() for line in down_content.split('\n') if line.strip()]
                                # Filter for likely gene names (all caps, contains letters/numbers)
                                down_genes = [line for line in lines 
                                            if re.match(r'^[A-Z][A-Z0-9_-]*$', line) and len(line) > 1]
                        
                        result['downregulated'] = down_genes
                    
            answer_content = result
            print(f"DEBUG: Final result: {answer_content}")

        return answer_content

    # Return None if no match or none of the keywords found
    return None


# --- Generation Helper ---
def run_generate(model, tokenizer, prompt, temperature, max_new_tokens, accelerator, num_return_sequences=1):
    """Runs model generation, handling potential errors and device placement."""
    # Use accelerator.unwrap_model to get the base model for generation
    unwrapped_model = accelerator.unwrap_model(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)

    # Move inputs to the device managed by accelerator for the current process
    input_ids = inputs.input_ids.to(accelerator.device)
    attention_mask = inputs.attention_mask.to(accelerator.device)

    # Ensure pad_token_id is set (often same as eos_token_id for decoder-only models)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine sampling parameters
    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = 50 # Common default, adjust if needed
        gen_kwargs["top_p"] = 0.95 # Common default, adjust if needed
    # else: use greedy decoding by default if do_sample is False

    try:
        with torch.no_grad(): # Ensure no gradients are computed during inference
            outputs = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        # Decode outputs, slicing off the input part
        output_sequences = outputs[:, input_ids.shape[-1]:]
        generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        # Strip leading/trailing whitespace from each generated text
        generated_texts = [text.strip() for text in generated_texts]
        
        return generated_texts if num_return_sequences > 1 else generated_texts[0]

    except Exception as e:
        accelerator.print(f"Error during generation on process {accelerator.process_index}: {e}")
        # Return placeholder error strings
        error_msg = "GENERATION_ERROR"
        return [error_msg] * num_return_sequences if num_return_sequences > 1 else error_msg



# --- Core Comparison Function ---
def compare_candidates(model, tokenizer, question_prompt, response_i, response_j, ktie, temperature, max_new_tokens, accelerator):
    """Performs ktie independent comparisons between two candidates."""
    i_wins = 0

    for attempt in range(ktie):
        history = "" # Reset history for each independent attempt
        current_context = {
            "question": question_prompt,
            "s1": response_i,
            "s2": response_j
        }
        winner = None # Track winner for this attempt

        for i, prompt_template in enumerate(comparison_prompts):
            try:
                if i > 0:
                    current_context["history"] = history
                formatted_prompt = prompt_template.format(**current_context)
            except KeyError as e:
                accelerator.print(f"Error formatting comparison prompt {i+1} on process {accelerator.process_index}: Missing key {e}")
                break # Break this attempt

            response = run_generate(model, tokenizer, formatted_prompt, temperature, max_new_tokens, accelerator)
            if response == "GENERATION_ERROR":
                 accelerator.print(f"Generation failed during comparison turn {i+1}, attempt {attempt+1}.")
                 break

            history += f"--- Turn {i+1} ---\n{response}\n\n"

            # --- Final Judgment Parsing (REVISED) ---
            if i == len(comparison_prompts) - 1:
                # Use regex to find all occurrences of '1' or '2' as standalone digits.
                # \b is a "word boundary" to prevent matching '1' in '10' or 'solution1'.
                matches = re.findall(r'\b[12]\b', response)

                if matches:
                    # Heuristic: If multiple numbers are found (e.g., "1 is good, but 2 is better"),
                    # assume the last one is the final decision.
                    final_choice = matches[-1]
                    if final_choice == '1':
                        winner = 'i'
                    elif final_choice == '2':
                        winner = 'j'
                else:
                    # No '1' or '2' found in the response.
                    accelerator.print(f"Warning (Process {accelerator.process_index}): Could not parse comparison judgment: '{response}'. No win awarded for this attempt.")
                    # winner remains None
                break # End of turns for this attempt

        # End of one comparison attempt
        if winner == 'i':
            i_wins += 1

    return i_wins


# --- Core Verification Function (Multi-Turn) ---
def verify_candidate(model, tokenizer, question_prompt, candidate_response, kverif, temperature, max_new_tokens, accelerator, task, pert, cell_type):
    """Performs kverif independent multi-turn verification attempts."""
    if task == "single_gene_prediction":
        # Use the single gene prediction prompts
        verification_prompts = verification_prompts_single_gene_prediction
        correct_count = 0

        for attempt in range(kverif):
            history = "" # Reset history for each independent attempt
            current_context = {
                "question": question_prompt,
                "candidate": candidate_response,
                "pert": pert,             
                "cell_type": cell_type 
            }
            is_correct_run = False
            # accelerator.print(f"Process {accelerator.process_index}, Verify Attempt {attempt+1}/{kverif}") # Verbose logging

            for i, prompt_template in enumerate(verification_prompts):
                # Format the prompt, adding history for turns > 0
                try:
                    if i > 0:
                        # Prevent history from becoming excessively long if needed
                        # (e.g., truncate history, summarize, etc. - not implemented here)
                        current_context["history"] = history
                    formatted_prompt = prompt_template.format(**current_context)
                except KeyError as e:
                    accelerator.print(f"Error formatting verification prompt {i+1} on process {accelerator.process_index}: Missing key {e}")
                    break # Break this attempt on formatting error

                # Generate response for this turn
                response = run_generate(model, tokenizer, formatted_prompt, temperature, max_new_tokens, accelerator)
                if response == "GENERATION_ERROR":
                    accelerator.print(f"Generation failed during verification turn {i+1}, attempt {attempt+1}.")
                    break # Stop this attempt if generation fails

                # Update history for the next turn
                # Combine prompt and response for context (or just response if prompts are implicit)
                history += f"--- Turn {i+1} ---\n{response}\n\n" # Store response as history context


                # --- Final Judgment Parsing ---
                if i == len(verification_prompts) - 1: # If this is the last turn
                    # TODO: Implement robust parsing logic for 'Correct' or 'Incorrect'
                    final_judgment = response.lower() # Use the raw response from the last turn
                    if "correct" in final_judgment and "incorrect" not in final_judgment:
                        is_correct_run = True
                    elif "incorrect" in final_judgment:
                        is_correct_run = False
                    else:
                        # Handle ambiguous cases or parsing failures
                        accelerator.print(f"Warning (Process {accelerator.process_index}): Could not parse verification judgment: '{response}'. Treating as Incorrect.")
                        is_correct_run = False
                    # accelerator.print(f"  Attempt {attempt+1} judged: {'Correct' if is_correct_run else 'Incorrect'}") # Verbose
                    break # End of turns for this attempt

            # End of one verification attempt (over all turns)
            if is_correct_run:
                correct_count += 1

        # Calculate average score
        verification_score = correct_count / kverif if kverif > 0 else 0.0
        return verification_score


    elif task == "direct_prediction":
        # Use the direct prediction prompts
        verification_prompts = verification_prompts_dirct_prediction
        total_scores = []

        for attempt in range(kverif):
            history = "" # Reset history for each independent attempt
            current_context = {
                "question": question_prompt,
                "candidate": candidate_response,
                "pert": pert,             
                "cell_type": cell_type 
            }
            # accelerator.print(f"Process {accelerator.process_index}, Verify Attempt {attempt+1}/{kverif}") # Verbose logging

            for i, prompt_template in enumerate(verification_prompts):
                # Format the prompt, adding history for turns > 0
                try:
                    if i > 0:
                        current_context["history"] = history
                    formatted_prompt = prompt_template.format(**current_context)
                except KeyError as e:
                    accelerator.print(f"Error formatting verification prompt {i+1} on process {accelerator.process_index}: Missing key {e}")
                    break # Break this attempt on formatting error

                # Generate response for this turn
                response = run_generate(model, tokenizer, formatted_prompt, temperature, max_new_tokens, accelerator)
                if response == "GENERATION_ERROR":
                    accelerator.print(f"Generation failed during verification turn {i+1}, attempt {attempt+1}.")
                    break # Stop this attempt if generation fails

                # Update history for the next turn
                history += f"--- Turn {i+1} ---\n{response}\n\n" # Store response as history context

                # --- Final Judgment Parsing for Scores ---
                if i == len(verification_prompts) - 1: # If this is the last turn
                    # Parse verification scores instead of binary judgment
                    scores_match = re.search(r'Upregulated:\s*(\d+).*?Downregulated:\s*(\d+)', response, re.IGNORECASE | re.DOTALL)
                    
                    if scores_match:
                        up_score = int(scores_match.group(1)) / 100.0
                        down_score = int(scores_match.group(2)) / 100.0
                        avg_score = (up_score + down_score) / 2.0
                        total_scores.append(avg_score)
                        # accelerator.print(f"  Attempt {attempt+1} scores: Up={up_score:.2f}, Down={down_score:.2f}, Avg={avg_score:.2f}") # Verbose
                    else:
                        # Handle parsing failures
                        accelerator.print(f"Warning (Process {accelerator.process_index}): Could not parse verification scores from: '{response[:100]}...'. Using score 0.0.")
                        total_scores.append(0.0)
                    break # End of turns for this attempt

        # Calculate average verification score across all attempts
        verification_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        return verification_score




# 
# Modified sampling_based_search to work with DiffExpressionDataset
def sampling_based_search_with_dataset(
    model,
    tokenizer,
    dataset_item,
    task,
    accelerator,
    kinf=5,
    kverif=3,
    ktie=5,
    sigma_inf=0.7,
    sigma_verif=0.0,
    sigma_tie=0.0,
    max_new_tokens_generate=512,
    max_new_tokens_verify=1024,
    max_new_tokens_compare=1024,
    score_tolerance=0.05
    ):
    """Modified sampling_based_search to work with DiffExpressionDataset items."""
    
    # Extract prompt from dataset item
    if isinstance(dataset_item["prompt"], list):
        # Format as a single string if it's a list of dicts
        prompt_parts = []
        for message in dataset_item["prompt"]:
            prompt_parts.append(f"{message['role']}:\n{message['content']}")
        question_prompt = "\n\n".join(prompt_parts)
    else:
        # Otherwise use as is
        question_prompt = dataset_item["prompt"]
    
    pert_gene = dataset_item.get("pert") # Use .get for safety if key might be missing
    cell_type_value = dataset_item.get("cell_type")
    
    # Run the original sampling_based_search function
    accelerator.print(f"--- Stage 1: Generating {kinf} Candidates ---")
    # Stage 1: Generate Responses
    candidate_responses = run_generate(
        model, tokenizer, question_prompt, sigma_inf, max_new_tokens_generate, accelerator, num_return_sequences=kinf
    )
    accelerator.print(f"Generated candidates: {candidate_responses}")
    
    # Filter out potential errors and duplicates
    valid_candidates = []
    seen_candidates = set()
    for r in candidate_responses:
        if r != "GENERATION_ERROR" and r not in seen_candidates:
            valid_candidates.append(r)
            seen_candidates.add(r)

    if not valid_candidates:
        accelerator.print("Error: No valid candidates generated.")
        return None, False, -1
    
    accelerator.print(f"Generated {len(valid_candidates)} unique valid candidates.")

    accelerator.print(f"\n--- Stage 2: Verifying {len(valid_candidates)} Candidates ({kverif} attempts each) ---")
    # Stage 2: Verify Responses
    verification_scores = {} # Store scores as {candidate_index: score}
    for i, response in enumerate(valid_candidates):
        score = verify_candidate(
            model, tokenizer, question_prompt, response, kverif, sigma_verif, max_new_tokens_verify, accelerator,
            task=task,
            pert=pert_gene,
            cell_type=cell_type_value
        )
        verification_scores[i] = score

    if not verification_scores:
        accelerator.print("Error: Verification failed for all candidates.")
        return None, False, -1

    # Find highest score
    max_score = max(verification_scores.values()) if verification_scores else -1.0
    accelerator.print(f"\nMax verification score found: {max_score:.4f}")

    # Gather best responses within tolerance
    SBest_indices = [
        i for i, score in verification_scores.items()
        if score >= max_score - score_tolerance
    ]
    accelerator.print(f"Found {len(SBest_indices)} candidates within tolerance ({score_tolerance*100}%). Indices: {SBest_indices}")

    # Check if tie-breaking is needed
    if len(SBest_indices) == 1:
        best_index = SBest_indices[0]
        accelerator.print(f"\n--- Single Best Candidate Found (Index {best_index}) ---")
        best_response = valid_candidates[best_index]
        # Extract predicted label
        predicted_label = extract_answer_from_response(best_response)
        # Compare with the true label
        if task == "single_gene_prediction":
            true_label = dataset_item["label"]
            is_correct = predicted_label == true_label
        elif task == "direct_prediction":
            true_label = dataset_item["raw_solution_lists"]
            f1_scores = calculate_gene_lists_f1(true_label, predicted_label)
            is_correct = f1_scores["average_f1"]

        return best_response, is_correct, predicted_label
    elif len(SBest_indices) == 0:
        # Fallback if tolerance somehow excluded all
        accelerator.print("Warning: No candidates within tolerance, selecting highest scored overall.")
        best_index = max(verification_scores, key=verification_scores.get)
        best_response = valid_candidates[best_index]
        predicted_label = extract_answer_from_response(best_response)
        if task == "single_gene_prediction":
            true_label = dataset_item["label"]
            is_correct = predicted_label == true_label
        elif task == "direct_prediction":
            true_label = dataset_item["raw_solution_lists"]
            f1_scores = calculate_gene_lists_f1(true_label, predicted_label)
            is_correct = f1_scores["average_f1"]

        return best_response, is_correct, predicted_label
    else:
        accelerator.print(f"\n--- Stage 3: Tie-Breaking Among {len(SBest_indices)} Candidates ({ktie} attempts per pair) ---")
        # Stage 3: Tie-Break via Pairwise Comparison
        SBest_candidates = {idx: valid_candidates[idx] for idx in SBest_indices}
        matchup_wins = defaultdict(int)

        # Generate pairs for comparison
        indices_to_compare = list(SBest_candidates.keys())
        pairs_to_compare = []
        for i in range(len(indices_to_compare)):
            for j in range(i + 1, len(indices_to_compare)):
                pairs_to_compare.append((indices_to_compare[i], indices_to_compare[j]))

        accelerator.print(f"Running {len(pairs_to_compare)} pairwise comparisons...")
        for idx_i, idx_j in pairs_to_compare:
            response_i = SBest_candidates[idx_i]
            response_j = SBest_candidates[idx_j]

            i_wins_count = compare_candidates(
                model, tokenizer, question_prompt, response_i, response_j, ktie, sigma_tie, max_new_tokens_compare, accelerator
            )
            j_wins_count = ktie - i_wins_count

            if i_wins_count > j_wins_count:
                matchup_wins[idx_i] += 1
            elif j_wins_count > i_wins_count:
                matchup_wins[idx_j] += 1

        # Determine the overall winner
        if not matchup_wins:
            accelerator.print("Warning: No decisive wins in tie-breaking. Selecting candidate with highest verification score.")
            best_index = max(SBest_candidates.keys(), key=lambda k: verification_scores[k])
        else:
            max_matchup_wins = max(matchup_wins.values())
            winners_with_max_wins = [idx for idx, wins in matchup_wins.items() if wins == max_matchup_wins]

            if len(winners_with_max_wins) == 1:
                best_index = winners_with_max_wins[0]
                accelerator.print(f"Tie-breaking winner: Candidate {best_index} with {matchup_wins[best_index]} matchup wins.")
            else:
                accelerator.print(f"Tie in matchup wins ({max_matchup_wins} wins) among indices {winners_with_max_wins}.")
                best_index = max(winners_with_max_wins, key=lambda k: verification_scores[k])
                accelerator.print(f"Final winner after tie-break: Candidate {best_index} (Verification Score: {verification_scores[best_index]:.4f})")
        
        best_response = SBest_candidates[best_index]
        predicted_label = extract_answer_from_response(best_response)
        if task == "single_gene_prediction":
            true_label = dataset_item["label"]
            is_correct = predicted_label == true_label
        elif task == "direct_prediction":
            true_label = dataset_item["raw_solution_lists"]
            f1_scores = calculate_gene_lists_f1(true_label, predicted_label)
            is_correct = f1_scores["average_f1"]

        return best_response, is_correct, predicted_label



def evaluate_dataset_with_search(csv_dir, model_name, output_dir, accelerator, task="single_gene_prediction", cell_type_split="default", list_format="bullet_list", **search_params):
    """Run the sampling-based search evaluation on a DiffExpressionDataset."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    # Load dataset - use 'test' split, modify params as needed
    if task == "single_gene_prediction":
        test_dataset = DiffExpressionDataset(
            csv_dir=csv_dir,
            split="test",
            tokenizer=tokenizer,
            prompt_mode="default", 
            test_split_cell_lines=cell_type_split,  
        )
    elif task == "direct_prediction":
        print("Using GeneRegulationListDataset for direct prediction task.")
        test_dataset = GeneRegulationListDataset(
            csv_dir=csv_dir,
            split="test",
            test_split_cell_lines=cell_type_split,
            list_format = list_format
        )
    # Get the number of processes and current process index
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    # Calculate which items this process should handle
    dataset_size = len(test_dataset)
    items_per_process = (dataset_size + num_processes - 1) // num_processes  # Ceiling division
    start_idx = process_index * items_per_process
    end_idx = min(start_idx + items_per_process, dataset_size)
    
    accelerator.print(f"Process {process_index}/{num_processes}: Processing items {start_idx} to {end_idx-1} out of {dataset_size}")
    
    # Initialize results storage
    results = []
    
    # Process assigned items
    for idx in tqdm(range(start_idx, end_idx), desc=f"Process {process_index}"):
        item = test_dataset[idx]
        
        accelerator.print(f"\n===== Process {process_index}: Starting Item {idx-start_idx+1}/{end_idx-start_idx} (Global Index {idx}) =====")
        if task == "single_gene_prediction":
            accelerator.print(f"Pert: {item['pert']}, Gene: {item['gene']}, Cell Type: {item['cell_type']}")
        else:
            accelerator.print(f"Pert: {item['pert']}, Cell Type: {item['cell_type']}")
        # Run the search algorithm
        best_response, is_correct, predicted_label = sampling_based_search_with_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_item=item,
            task=task,
            accelerator=accelerator,
            **search_params
        )
        if task == "single_gene_prediction":
            # Store results
            results.append({
                "pert": item["pert"],
                "gene": item["gene"],
                "cell_type": item["cell_type"],
                "true_label": item["label"],
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "best_response": best_response
            })
            accelerator.print(f"Evaluation: True label={item['label']}, Predicted label={predicted_label}, Correct={is_correct}")

        else:
            results.append({
                "pert": item["pert"],
                "cell_type": item["cell_type"],
                "true_label": item["raw_solution_lists"],
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "best_response": best_response
            })
            accelerator.print(f"Evaluation: True labels={item['raw_solution_lists']}, Predicted labels={predicted_label}, Correct={is_correct}")
        
        accelerator.print(f"===== Process {process_index}: Finished Item {idx-start_idx+1}/{end_idx-start_idx} (Global Index {idx}) =====")
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    # Gather results from all processes
    all_results = accelerator.gather_object(results)
    
    # Main process combines and saves results
    if accelerator.is_main_process:
        combined_results = []
        for proc_results in all_results:
            combined_results.extend(proc_results)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(combined_results)
        
        if task == "single_gene_prediction":

            # Calculate overall accuracy
            accuracy = results_df["is_correct"].mean()
            print(f"\nOverall accuracy: {accuracy:.4f}")
            
            # Calculate accuracy per label
            label_accuracy = results_df.groupby("true_label")["is_correct"].mean()
            print("\nAccuracy per label:")
            for label, acc in label_accuracy.items():
                label_name = {0: "not differentially expressed", 1: "downregulated", 2: "upregulated"}[label]
                print(f"  {label} ({label_name}): {acc:.4f}")
            
            # Calculate confusion matrix
            confusion = pd.crosstab(
                results_df["true_label"], 
                results_df["predicted_label"],
                rownames=["True"],
                colnames=["Predicted"]
            )
            print("\nConfusion Matrix:")
            print(confusion)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filenames
            model_short_name = model_name.split('/')[-1]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            results_filename = f"{task}_{model_short_name}_results_{timestamp}.csv"
            summary_filename = f"{task}_{model_short_name}_summary_{timestamp}.txt"
            
            # Save results to file
            results_file_path = output_path / results_filename
            results_df.to_csv(results_file_path, index=False)
            print(f"\nDetailed results saved to {results_file_path}")
            
            # Save summary statistics
            summary_file_path = output_path / summary_filename
            with open(summary_file_path, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {csv_dir}\n")
                f.write(f"Evaluation timestamp: {timestamp}\n")
                f.write(f"Search parameters: {search_params}\n\n")
                f.write(f"Total samples: {len(results_df)}\n")
                f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
                f.write("Accuracy per label:\n")
                for label, acc in label_accuracy.items():
                    label_name = {0: "not differentially expressed", 1: "downregulated", 2: "upregulated"}[label]
                    f.write(f"  {label} ({label_name}): {acc:.4f}\n")
                f.write("\nConfusion Matrix:\n")
                f.write(confusion.to_string())
            
            print(f"Summary statistics saved to {summary_file_path}")
        
        elif task == "direct_prediction":
            
            # Use F1 scores instead
            avg_f1 = results_df["average_f1"].mean()
            avg_up_f1 = results_df["upregulated_f1"].mean()
            avg_down_f1 = results_df["downregulated_f1"].mean()
            
            print(f"\nOverall Average F1: {avg_f1:.4f}")
            print(f"Average Upregulated F1: {avg_up_f1:.4f}")
            print(f"Average Downregulated F1: {avg_down_f1:.4f}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filenames
            model_short_name = model_name.split('/')[-1]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            results_filename = f"{task}_{model_short_name}_results_{timestamp}.csv"
            summary_filename = f"{task}_{model_short_name}_summary_{timestamp}.txt"
            
            # Save results to file
            results_file_path = output_path / results_filename
            results_df.to_csv(results_file_path, index=False)
            print(f"\nDetailed results saved to {results_file_path}")
            
            # Save summary statistics
            summary_file_path = output_path / summary_filename

            with open(summary_file_path, "w") as f: 

                f.write(f"Model: {model_name}\n") 
                f.write(f"Dataset: {csv_dir}\n") 
                f.write(f"Task: {task}\n") 
                f.write(f"Evaluation timestamp: {timestamp}\n")
                f.write(f"Search parameters: {search_params}\n\n") 
                f.write(f"Total samples: {len(results_df)}\n")        
                f.write(f"Overall Average F1: {avg_f1:.4f}\n")
                f.write(f"Average Upregulated F1: {avg_up_f1:.4f}\n")
                f.write(f"Average Downregulated F1: {avg_down_f1:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gene regulation prediction using sampling-based search")
    parser.add_argument("--csv_dir", required=True, help="Directory containing differential expression CSV files")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name or path")
    parser.add_argument("--output_dir", required=True, help="Directory to save results and summary files")
    parser.add_argument("--kinf", type=int, default=5, help="Number of candidate responses to generate")
    parser.add_argument("--kverif", type=int, default=3, help="Number of verification attempts per candidate")
    parser.add_argument("--ktie", type=int, default=5, help="Number of comparison attempts for tie-breaking")
    parser.add_argument("--sigma_inf", type=float, default=0.7, help="Temperature for candidate generation")
    parser.add_argument("--sigma_verif", type=float, default=0.0, help="Temperature for verification")
    parser.add_argument("--sigma_tie", type=float, default=0.0, help="Temperature for tie-breaking")
    parser.add_argument("--max_tokens_gen", type=int, default=8192, help="Max tokens for generation")
    parser.add_argument("--max_tokens_verify", type=int, default=8192, help="Max tokens for verification")
    parser.add_argument("--max_tokens_compare", type=int, default=8192, help="Max tokens for comparison")
    parser.add_argument("--score_tolerance", type=float, default=0.05, help="Tolerance for verification scores")
    parser.add_argument("--task", type=str, default="single_gene_prediction", help="Task type: single_gene_prediction or direct_prediction")
    parser.add_argument("--cell_type_split", type=str, default="default", help="Cell type split for dataset")
    parser.add_argument("--list_format", type=str, default="bullet_list", help="List format for direct prediction task")
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Run evaluation
    evaluate_dataset_with_search(
        csv_dir=args.csv_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        accelerator=accelerator,
        kinf=args.kinf,
        kverif=args.kverif,
        ktie=args.ktie,
        sigma_inf=args.sigma_inf,
        sigma_verif=args.sigma_verif,
        sigma_tie=args.sigma_tie,
        max_new_tokens_generate=args.max_tokens_gen,
        max_new_tokens_verify=args.max_tokens_verify,
        max_new_tokens_compare=args.max_tokens_compare,
        score_tolerance=args.score_tolerance,
        task=args.task,
        cell_type_split=args.cell_type_split,
        list_format=args.list_format
    )

# --- Main Algorithm 1 Implementation ---
# def sampling_based_search(
#     model,
#     tokenizer,
#     question_prompt,
#     dataset_item,
#     accelerator,
#     kinf=200,
#     kverif=50,
#     ktie=100,
#     sigma_inf=1.0,
#     sigma_verif=0.0,
#     sigma_tie=0.0,
#     max_new_tokens_generate=8192,
#     max_new_tokens_verify=8192,
#     max_new_tokens_compare=8192,
#     score_tolerance=0.05
#     ):
#     """Implements Algorithm 1: Sample, Scrutinize and Scale for a single question."""

#     # Use accelerator.print for logging in distributed setup
#     accelerator.print(f"--- Stage 1: Generating {kinf} Candidates ---")
#     # Stage 1: Generate Responses
#     candidate_responses = run_generate(
#         model, tokenizer, question_prompt, sigma_inf, max_new_tokens_generate, accelerator, num_return_sequences=kinf
#     )
#     #DEBUG: Check generated candidates
#     accelerator.print(f"Generated candidates: {candidate_responses}") # Verbose

    
#     # Filter out potential errors and duplicates
#     valid_candidates = []
#     seen_candidates = set()
#     for r in candidate_responses:
#         if r != "GENERATION_ERROR" and r not in seen_candidates:
#             valid_candidates.append(r)
#             seen_candidates.add(r)

#     if not valid_candidates:
#         accelerator.print("Error: No valid candidates generated.")
#         return None # Or return an error indicator
#     accelerator.print(f"Generated {len(valid_candidates)} unique valid candidates.")
#     # Optional: Limit number of candidates if too many were generated after filtering
#     # valid_candidates = valid_candidates[:kinf]


#     accelerator.print(f"\n--- Stage 2: Verifying {len(valid_candidates)} Candidates ({kverif} attempts each) ---")
#     # Stage 2: Verify Responses
#     verification_scores = {} # Store scores as {candidate_index: score}
#     for i, response in enumerate(valid_candidates):
#         # accelerator.print(f"Verifying candidate {i+1}/{len(valid_candidates)}...") # Can be too verbose
#         score = verify_candidate(
#             model, tokenizer, question_prompt, response, kverif, sigma_verif, max_new_tokens_verify, accelerator
#         )
#         verification_scores[i] = score
#         # accelerator.print(f"Candidate {i+1} score: {score:.4f}") # Can be too verbose

#     if not verification_scores:
#          accelerator.print("Error: Verification failed for all candidates.")
#          return None

#     # Find highest score
#     max_score = max(verification_scores.values()) if verification_scores else -1.0
#     accelerator.print(f"\nMax verification score found: {max_score:.4f}")

#     # Gather best responses within tolerance (SBest in Algorithm 1)
#     SBest_indices = [
#         i for i, score in verification_scores.items()
#         if score >= max_score - score_tolerance
#     ]
#     accelerator.print(f"Found {len(SBest_indices)} candidates within tolerance ({score_tolerance*100}%). Indices: {SBest_indices}")

#     # Check if tie-breaking is needed
#     if len(SBest_indices) == 1:
#         best_index = SBest_indices[0]
#         accelerator.print(f"\n--- Single Best Candidate Found (Index {best_index}) ---")
#         return valid_candidates[best_index]
#     elif len(SBest_indices) == 0:
#          # Fallback if tolerance somehow excluded all, select absolute best
#          accelerator.print("Warning: No candidates within tolerance, selecting highest scored overall.")
#          best_index = max(verification_scores, key=verification_scores.get)
#          return valid_candidates[best_index]
#     else:
#         accelerator.print(f"\n--- Stage 3: Tie-Breaking Among {len(SBest_indices)} Candidates ({ktie} attempts per pair) ---")
#         # Stage 3: Tie-Break via Pairwise Comparison
#         SBest_candidates = {idx: valid_candidates[idx] for idx in SBest_indices}
#         matchup_wins = defaultdict(int) # Store wins for each candidate index

#         # Generate pairs for comparison
#         indices_to_compare = list(SBest_candidates.keys())
#         pairs_to_compare = []
#         for i in range(len(indices_to_compare)):
#             for j in range(i + 1, len(indices_to_compare)):
#                 pairs_to_compare.append((indices_to_compare[i], indices_to_compare[j]))

#         accelerator.print(f"Running {len(pairs_to_compare)} pairwise comparisons...")
#         for idx_i, idx_j in pairs_to_compare:
#             response_i = SBest_candidates[idx_i]
#             response_j = SBest_candidates[idx_j]

#             # accelerator.print(f"Comparing candidate {idx_i} vs {idx_j}...") # Verbose
#             i_wins_count = compare_candidates(
#                 model, tokenizer, question_prompt, response_i, response_j, ktie, sigma_tie, max_new_tokens_compare, accelerator
#             )
#             j_wins_count = ktie - i_wins_count # Total comparisons = ktie
#             # accelerator.print(f"Result: C{idx_i} wins: {i_wins_count}, C{idx_j} wins: {j_wins_count}") # Verbose

#             # Award points based on wins in this matchup
#             if i_wins_count > j_wins_count:
#                 matchup_wins[idx_i] += 1
#             elif j_wins_count > i_wins_count:
#                 matchup_wins[idx_j] += 1
#             # else: Tie in this specific matchup, no points awarded

#         # Determine the overall winner based on total matchup wins
#         if not matchup_wins:
#              accelerator.print("Warning: No decisive wins in tie-breaking. Selecting candidate with highest verification score among ties.")
#              # Fallback: select the one with the absolute highest verification score among the tied group
#              best_index = max(SBest_candidates.keys(), key=lambda k: verification_scores[k])
#         else:
#             # Find the index with the maximum number of matchup wins
#             # Handle potential ties in *matchup wins* by falling back to verification score
#             max_matchup_wins = max(matchup_wins.values())
#             winners_with_max_wins = [idx for idx, wins in matchup_wins.items() if wins == max_matchup_wins]

#             if len(winners_with_max_wins) == 1:
#                 best_index = winners_with_max_wins[0]
#                 accelerator.print(f"Tie-breaking winner: Candidate {best_index} with {matchup_wins[best_index]} matchup wins.")
#             else:
#                 accelerator.print(f"Tie in matchup wins ({max_matchup_wins} wins) among indices {winners_with_max_wins}. Breaking tie with verification score.")
#                 # Select the one with the highest verification score among those tied in matchup wins
#                 best_index = max(winners_with_max_wins, key=lambda k: verification_scores[k])
#                 accelerator.print(f"Final winner after tie-break: Candidate {best_index} (Verification Score: {verification_scores[best_index]:.4f})")


#         return SBest_candidates[best_index]



# # --- Main Execution Block ---
# if __name__ == '__main__':
#     # Initialize Accelerator FIRST
#     accelerator = Accelerator()

#     # --- Configuration ---
#     # Consider using argparse for better configuration management
#     model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Or your preferred model
#     # Use smaller values for quick testing, increase for full runs matching paper
#     PARAM_KINF = 5     # Paper: 200+
#     PARAM_KVERIF = 3   # Paper: 50+
#     PARAM_KTIE = 5     # Paper: 100
#     PARAM_SIGMA_INF = 0.7 # Temperature for diverse candidate generation
#     PARAM_SIGMA_VERIF = 0.0 # Often deterministic (0) for verification consistency
#     PARAM_SIGMA_TIE = 0.0 # Often deterministic (0) for tie-breaking consistency
#     PARAM_MAX_TOKENS_GEN = 512
#     PARAM_MAX_TOKENS_VERIFY = 1024 # Verification turns can be long
#     PARAM_MAX_TOKENS_COMPARE = 1024 # Comparison turns can be long
#     PARAM_SCORE_TOLERANCE = 0.05

#     # --- Model Loading ---
#     accelerator.print(f"Loading model: {model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
#         trust_remote_code=True,
#         # Add low_cpu_mem_usage=True if memory is tight during loading
#         # low_cpu_mem_usage=True
#     )

#     # Prepare model and tokenizer WITH Accelerator
#     # This handles device placement and potential model sharding/parallelism
#     model, tokenizer = accelerator.prepare(model, tokenizer)
#     accelerator.print("Model and tokenizer prepared.")

#     # --- Load and Distribute Data (List of Questions) ---
#     # Replace this with loading your actual dataset

    
    
#     # all_math_questions = [
#     #     "What is 2+2?",
#     #     "What is 5*3?",
#     #     "Solve for x: 2x = 10",
#     #     "What is the square root of 64?",
#     #     "If a train travels at 60 mph for 3 hours, how far does it travel?",
#     #     "Simplify the expression: 3 * (4 + 2)",
#     #     "What is 15% of 200?",
#     #     "Find the area of a rectangle with length 5 and width 8."
#     # ]
#     num_questions = len(all_math_questions)
#     num_processes = accelerator.num_processes
#     process_index = accelerator.process_index

#     # Calculate data split for this process
#     questions_per_process = math.ceil(num_questions / num_processes)
#     start_index = process_index * questions_per_process
#     end_index = min(start_index + questions_per_process, num_questions)
#     questions_for_this_process = all_math_questions[start_index:end_index]

#     if not questions_for_this_process:
#         accelerator.print(f"Process {process_index}/{num_processes}: No questions assigned.")
#     else:
#         accelerator.print(f"Process {process_index}/{num_processes}: Processing {len(questions_for_this_process)} questions (Indices {start_index} to {end_index-1}).")

#     # --- Process Assigned Questions ---
#     results_this_process = {} # Store {question: best_response}
#     for i, question in enumerate(questions_for_this_process):
#         global_index = start_index + i
#         accelerator.print(f"\n===== Process {process_index}: Starting Question {i+1}/{len(questions_for_this_process)} (Global Index {global_index}) =====")
#         accelerator.print(f"Question: {question}")

#         best_response = sampling_based_search(
#             model=model,
#             tokenizer=tokenizer,
#             question_prompt=question,
#             accelerator=accelerator,
#             kinf=PARAM_KINF,
#             kverif=PARAM_KVERIF,
#             ktie=PARAM_KTIE,
#             sigma_inf=PARAM_SIGMA_INF,
#             sigma_verif=PARAM_SIGMA_VERIF,
#             sigma_tie=PARAM_SIGMA_TIE,
#             max_new_tokens_generate=PARAM_MAX_TOKENS_GEN,
#             max_new_tokens_verify=PARAM_MAX_TOKENS_VERIFY,
#             max_new_tokens_compare=PARAM_MAX_TOKENS_COMPARE,
#             score_tolerance=PARAM_SCORE_TOLERANCE
#         )

#         results_this_process[question] = best_response if best_response else "SEARCH_FAILED"
#         accelerator.print(f"===== Process {process_index}: Finished Question {i+1}/{len(questions_for_this_process)} (Global Index {global_index}) =====")


#     # --- Gather and Finalize Results ---
#     accelerator.wait_for_everyone() # Important: ensure all processes finish before gathering
#     accelerator.print(f"Process {process_index} finished processing. Waiting to gather results...")

#     # Gather results from all processes onto the main process
#     gathered_results_list = gather_object(results_this_process)

#     if accelerator.is_main_process:
#         print("\n===================================")
#         print("Gathering results on main process...")
#         print("===================================")

#         final_results = {}
#         if gathered_results_list: # Check if list is not empty
#              for process_results in gathered_results_list:
#                  if isinstance(process_results, dict): # Ensure it's a dictionary
#                      final_results.update(process_results)
#                  else:
#                      print(f"Warning: Received non-dict object during gathering: {type(process_results)}")

#         print(f"\nTotal questions processed across all processes: {len(final_results)}")

#         # Optional: Save results to a file
#         output_filename = "final_search_results.txt" # Or .json
#         print(f"Saving results to {output_filename}...")
#         with open(output_filename, "w", encoding="utf-8") as f:
#              f.write("=" * 30 + " FINAL RESULTS " + "=" * 30 + "\n\n")
#              # Sort by original order if needed, although dict order is stable in Python 3.7+
#              # sorted_questions = sorted(final_results.keys(), key=lambda q: all_math_questions.index(q))
#              # for question in sorted_questions:
#              for question, response in final_results.items():
#                  f.write(f"Question: {question}\n")
#                  f.write(f"Best Response:\n{response}\n")
#                  f.write("-" * 70 + "\n")
#         print(f"Results saved.")

#         print("\n--- Main process finished ---")