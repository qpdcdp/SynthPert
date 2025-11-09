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
        # gen_kwargs["top_k"] = 50 # Common default, adjust if needed
        # gen_kwargs["top_p"] = 0.95 # Common default, adjust if needed
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
    """
    Performs kverif independent multi-turn verification attempts and returns detailed logs.
    Returns:
        tuple: (average_score, verification_logs)
        - average_score (float): The mean score across all kverif attempts.
        - verification_logs (list): A list of dicts, one for each attempt, containing the final response and parsed score.
    """
    verification_logs = []
    total_scores = []
    
    verification_prompts = (
        verification_prompts_single_gene_prediction
        if task == "single_gene_prediction"
        else verification_prompts_dirct_prediction
    )

    for attempt in range(kverif):
        history = ""
        current_context = {
            "question": question_prompt,
            "candidate": candidate_response,
            "pert": pert,
            "cell_type": cell_type,
        }
        
        final_response_text = "VERIFICATION_FAILED"
        parsed_score = 0.0

        for i, prompt_template in enumerate(verification_prompts):
            try:
                if i > 0:
                    current_context["history"] = history
                formatted_prompt = prompt_template.format(**current_context)
            except KeyError as e:
                accelerator.print(f"Error formatting verification prompt {i+1}: Missing key {e}")
                break

            response = run_generate(model, tokenizer, formatted_prompt, temperature, max_new_tokens, accelerator)
            if response == "GENERATION_ERROR":
                break

            history += f"--- Turn {i+1} ---\n{response}\n\n"

            if i == len(verification_prompts) - 1:
                final_response_text = response # This is the final judgment response
                if task == "single_gene_prediction":
                    if "correct" in response.lower() and "incorrect" not in response.lower():
                        parsed_score = 1.0
                    else:
                        parsed_score = 0.0
                elif task == "direct_prediction":
                    scores_match = re.search(r'Upregulated:\s*(\d+).*?Downregulated:\s*(\d+)', response, re.IGNORECASE | re.DOTALL)
                    if scores_match:
                        up_score = int(scores_match.group(1)) / 100.0
                        down_score = int(scores_match.group(2)) / 100.0
                        parsed_score = (up_score + down_score) / 2.0
                    else:
                        parsed_score = 0.0
                
                total_scores.append(parsed_score)

        verification_logs.append({
            "attempt_num": attempt + 1,
            "final_verification_response": final_response_text,
            "parsed_score": parsed_score,
        })

    average_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    return average_score, verification_logs


# --- Core Comparison Function ---
# (Assuming compare_candidates is defined as in your original script)
# ...


# --- Core Search Function with Detailed Logging ---
def sampling_based_search_with_detailed_logging(
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
    """
    Runs the full search process and returns a comprehensive log dictionary,
    while printing detailed live analysis to the console.
    """
    # Helper function for printing the ranking table
    def print_candidate_summary_table(candidates, verification_scores, ground_truth, task):
        """
        Calculates metrics for all candidates, sorts them, and prints a comparative table.
        """
        candidate_data = []
        for i, response_text in enumerate(candidates):
            verif_score = verification_scores.get(i, 0.0)
            predicted_label = extract_answer_from_response(response_text)
            
            gt_score = 0.0
            if task == "single_gene_prediction":
                gt_score = 1.0 if (predicted_label == ground_truth) else 0.0
            elif task == "direct_prediction":
                f1_scores = calculate_gene_lists_f1(ground_truth, predicted_label)
                gt_score = f1_scores.get("average_f1", 0.0)
                
            candidate_data.append({
                "index": i,
                "verif_score": verif_score,
                "gt_score": gt_score
            })
        
        # Sort by verification score (descending) to get the model's ranking
        sorted_by_verif = sorted(candidate_data, key=lambda x: x["verif_score"], reverse=True)
        
        # Sort by ground truth score (descending) to get the true ranking
        sorted_by_gt = sorted(candidate_data, key=lambda x: x["gt_score"], reverse=True)
        
        # Create a map for quick lookup of true rank
        gt_rank_map = {item['index']: rank + 1 for rank, item in enumerate(sorted_by_gt)}

        accelerator.print("\n--- Candidate Ranking Comparison ---")
        header = f"{'Rank (by Verif)':<16} | {'Rank (by GT)':<14} | {'Cand. Index':<12} | {'Verif. Score':<14} | {'GT Score (F1/Acc)':<20}"
        accelerator.print(header)
        accelerator.print("-" * len(header))
        
        for i, verif_rank_item in enumerate(sorted_by_verif):
            cand_index = verif_rank_item['index']
            gt_rank = gt_rank_map[cand_index]
            
            line = (
                f"{i+1:<16} | "
                f"{gt_rank:<14} | "
                f"{cand_index:<12} | "
                f"{verif_rank_item['verif_score']:<14.3f} | "
                f"{verif_rank_item['gt_score']:<20.3f}"
            )
            accelerator.print(line)
            
        # Highlight the best candidate selected by the model
        best_verif_candidate_idx = sorted_by_verif[0]['index']
        accelerator.print(f"\nModel's Choice (highest verif score): Candidate {best_verif_candidate_idx}")
        
        # Highlight the actual best candidate
        best_gt_candidate_idx = sorted_by_gt[0]['index']
        accelerator.print(f"True Best (highest GT score):       Candidate {best_gt_candidate_idx}")
        
        if best_verif_candidate_idx == best_gt_candidate_idx:
            accelerator.print(">> RANKING SUCCESS: Model correctly identified the best candidate.")
        else:
            accelerator.print(">> RANKING MISMATCH: Model did NOT identify the best candidate.")

    # Initialize the log object for this specific dataset item
    item_log = {
        "item_details": {
            "pert": dataset_item.get("pert"),
            "cell_type": dataset_item.get("cell_type"),
            "gene": dataset_item.get("gene"),
            "ground_truth_label": dataset_item.get("label") if task == "single_gene_prediction" else dataset_item.get("raw_solution_lists")
        },
        "generations": [],
        "verifications": {},
        "tie_break": None,
        "final_result": {}
    }

    # Extract prompt
    if isinstance(dataset_item["prompt"], list):
        prompt_parts = [f"{m['role']}:\n{m['content']}" for m in dataset_item["prompt"]]
        question_prompt = "\n\n".join(prompt_parts)
    else:
        question_prompt = dataset_item["prompt"]

    # --- Stage 1: Generation ---
    accelerator.print(f"--- Stage 1: Generating {kinf} Candidates ---")
    candidate_responses = run_generate(
        model, tokenizer, question_prompt, sigma_inf, max_new_tokens_generate, accelerator, num_return_sequences=kinf
    )
    
    valid_candidates = [r for r in candidate_responses if r != "GENERATION_ERROR"]
    
    for i, response in enumerate(valid_candidates):
        item_log["generations"].append({
            "candidate_index": i,
            "response_text": response
        })
    
    if not valid_candidates:
        item_log["final_result"]["error"] = "No valid candidates generated."
        return item_log

    # --- Stage 2: Verification ---
    accelerator.print(f"\n--- Stage 2: Verifying {len(valid_candidates)} Candidates ({kverif} attempts each) ---")
    verification_scores = {}
    for i, response in enumerate(valid_candidates):
        score, verif_details = verify_candidate(
            model, tokenizer, question_prompt, response, kverif, sigma_verif, max_new_tokens_verify, accelerator,
            task=task, pert=dataset_item.get("pert"), cell_type=dataset_item.get("cell_type")
        )
        verification_scores[i] = score
        item_log["verifications"][f"candidate_{i}"] = {
            "average_score": score,
            "attempts": verif_details
        }

    # Live console analysis
    ground_truth_label = item_log["item_details"]["ground_truth_label"]
    print_candidate_summary_table(
        candidates=valid_candidates,
        verification_scores=verification_scores,
        ground_truth=ground_truth_label,
        task=task
    )
    
    max_score = max(verification_scores.values()) if verification_scores else -1.0
    accelerator.print(f"\nMax verification score identified: {max_score:.3f}")

    # --- Stage 3: Selection & Tie-Breaking ---
    SBest_indices = [i for i, score in verification_scores.items() if score >= max_score - score_tolerance]
    accelerator.print(f"Found {len(SBest_indices)} candidates for tie-breaking. Indices: {SBest_indices}")
    
    best_index = -1
    if len(SBest_indices) == 1:
        best_index = SBest_indices[0]
        accelerator.print(f"\n--- Single Best Candidate Found (Index {best_index}) ---")
    elif len(SBest_indices) == 0 and verification_scores:
        best_index = max(verification_scores, key=verification_scores.get)
        accelerator.print("\n--- Warning: No candidates in tolerance, selecting highest scored overall. ---")
    elif len(SBest_indices) > 1:
        accelerator.print(f"\n--- Stage 3: Tie-Breaking Among {len(SBest_indices)} Candidates ---")
        item_log["tie_break"] = {"candidates_in_tie_break": SBest_indices, "matchups": []}
        
        SBest_candidates = {idx: valid_candidates[idx] for idx in SBest_indices}
        matchup_wins = defaultdict(int)

        indices_to_compare = list(SBest_candidates.keys())
        pairs_to_compare = [(indices_to_compare[i], indices_to_compare[j]) for i in range(len(indices_to_compare)) for j in range(i + 1, len(indices_to_compare))]

        for idx_i, idx_j in pairs_to_compare:
            i_wins_count = compare_candidates(model, tokenizer, question_prompt, SBest_candidates[idx_i], SBest_candidates[idx_j], ktie, sigma_tie, max_new_tokens_compare, accelerator)
            j_wins_count = ktie - i_wins_count
            
            winner_idx = -1
            if i_wins_count > j_wins_count:
                matchup_wins[idx_i] += 1
                winner_idx = idx_i
            elif j_wins_count > i_wins_count:
                matchup_wins[idx_j] += 1
                winner_idx = idx_j
            
            item_log["tie_break"]["matchups"].append({
                "pair": [idx_i, idx_j], "wins": {f"candidate_{idx_i}": i_wins_count, f"candidate_{idx_j}": j_wins_count}, "winner": f"candidate_{winner_idx}" if winner_idx != -1 else "tie"
            })

        if not matchup_wins:
            best_index = max(SBest_candidates.keys(), key=lambda k: verification_scores[k])
        else:
            max_wins = max(matchup_wins.values())
            winners = [idx for idx, wins in matchup_wins.items() if wins == max_wins]
            best_index = max(winners, key=lambda k: verification_scores[k])
        
        item_log["tie_break"]["final_winner_index"] = best_index
    
    # --- Final Result Collation ---
    if best_index == -1 and valid_candidates:
        accelerator.print("Error: Could not determine a best candidate. Defaulting to first valid candidate.")
        best_index = 0

    if best_index != -1:
        best_response = valid_candidates[best_index]
        predicted_label = extract_answer_from_response(best_response)
        
        is_correct_metric = None
        if task == "single_gene_prediction":
            is_correct_metric = (predicted_label == item_log["item_details"]["ground_truth_label"])
        elif task == "direct_prediction":
            f1_scores = calculate_gene_lists_f1(item_log["item_details"]["ground_truth_label"], predicted_label)
            is_correct_metric = f1_scores
        
        item_log["final_result"] = {
            "best_candidate_index": best_index,
            "best_response_text": best_response,
            "predicted_label": predicted_label,
            "ground_truth_comparison_metric": is_correct_metric
        }
    else:
        item_log["final_result"]["error"] = "No valid candidates to process."

    return item_log


# --- Main Evaluation Function ---
def evaluate_dataset_with_search(csv_dir, model_name, output_dir, accelerator, task="single_gene_prediction", cell_type_split="default", list_format="bullet_list", **search_params):
    """Run the sampling-based search evaluation on a dataset, saving detailed and summary logs."""
    
    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, trust_remote_code=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Load dataset
    if task == "single_gene_prediction":
        test_dataset = DiffExpressionDataset(csv_dir=csv_dir, split="test", tokenizer=tokenizer, prompt_mode="default", test_split_cell_lines=cell_type_split)
    else:
        test_dataset = GeneRegulationListDataset(csv_dir=csv_dir, split="test", test_split_cell_lines=cell_type_split, list_format=list_format)
    
    # Distribute work
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    dataset_size = len(test_dataset)
    items_per_process = (dataset_size + num_processes - 1) // num_processes
    start_idx = process_index * items_per_process
    end_idx = min(start_idx + items_per_process, dataset_size)
    
    all_item_logs = []
    for idx in tqdm(range(start_idx, end_idx), desc=f"Process {process_index}"):
        item = test_dataset[idx]
        accelerator.print(f"\n===== Process {process_index}: Starting Item {idx} =====")
        
        item_log = sampling_based_search_with_detailed_logging(
            model=model, tokenizer=tokenizer, dataset_item=item, task=task, accelerator=accelerator, **search_params
        )
        all_item_logs.append(item_log)
        
        accelerator.print(f"===== Process {process_index}: Finished Item {idx} =====")

    accelerator.wait_for_everyone()
    gathered_logs = accelerator.gather_object(all_item_logs)

    if accelerator.is_main_process:
        final_logs = [item for sublist in gathered_logs for item in sublist]

        # Prepare filenames
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_short_name = model_name.split('/')[-1]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        
        summary_results_csv = output_path / f"{task}_{model_short_name}_summary_{timestamp}.csv"
        detailed_log_jsonl = output_path / f"{task}_{model_short_name}_detailed_log_{timestamp}.jsonl"
        summary_report_txt = output_path / f"{task}_{model_short_name}_report_{timestamp}.txt"

        # 1. Save Detailed JSONL Log
        print(f"\nSaving detailed logs to {detailed_log_jsonl}...")
        with open(detailed_log_jsonl, 'w') as f:
            for log_entry in final_logs:
                f.write(json.dumps(log_entry) + '\n')

        # 2. Create and Save Summary CSV
        summary_data = []
        for log in final_logs:
            res = {
                "pert": log["item_details"]["pert"], "cell_type": log["item_details"]["cell_type"], "gene": log["item_details"]["gene"],
                "ground_truth": log["item_details"]["ground_truth_label"], "predicted_label": log["final_result"].get("predicted_label"),
            }
            if task == "single_gene_prediction":
                res["is_correct"] = log["final_result"].get("ground_truth_comparison_metric")
            elif task == "direct_prediction":
                f1_scores = log["final_result"].get("ground_truth_comparison_metric", {})
                res["upregulated_f1"] = f1_scores.get("upregulated_f1")
                res["downregulated_f1"] = f1_scores.get("downregulated_f1")
                res["average_f1"] = f1_scores.get("average_f1")
            summary_data.append(res)
        
        results_df = pd.DataFrame(summary_data)
        results_df.to_csv(summary_results_csv, index=False)
        print(f"Summary results saved to {summary_results_csv}")

        # 3. Generate and Save Final Text Report
        with open(summary_report_txt, "w") as f:
            f.write(f"Model: {model_name}\nDataset: {csv_dir}\nTask: {task}\nTimestamp: {timestamp}\n")
            f.write(f"Search Parameters: {search_params}\n\nTotal Samples Evaluated: {len(results_df)}\n\n")

            if task == "single_gene_prediction":
                accuracy = results_df["is_correct"].mean() if not results_df.empty else 0
                f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            elif task == "direct_prediction":
                avg_f1 = results_df["average_f1"].mean() if not results_df.empty else 0
                avg_up_f1 = results_df["upregulated_f1"].mean() if not results_df.empty else 0
                avg_down_f1 = results_df["downregulated_f1"].mean() if not results_df.empty else 0
                f.write(f"Overall Average F1 Score: {avg_f1:.4f}\n")
                f.write(f"  - Average Upregulated F1: {avg_up_f1:.4f}\n")
                f.write(f"  - Average Downregulated F1: {avg_down_f1:.4f}\n")
        print(f"Final report saved to {summary_report_txt}")


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
    
    accelerator = Accelerator()
    
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
