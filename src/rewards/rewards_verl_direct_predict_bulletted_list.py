import torch
import warnings
import numpy as np
import logging
import re


# extract content function

def extract_content(text, tag):
    """Extracts content between the first occurrence of <tag>...</tag>."""
    if not isinstance(text, str) or not text:
        return None
        
    start_tag = f"<{tag}"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
         return None

    pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def parse_gene_lists(answer_content):
    """
    Parses the new format with UPREGULATED_GENES and DOWNREGULATED_GENES sections.
    Returns a dictionary with 'upregulated' and 'downregulated' keys, each containing a list of genes.
    """
    if not answer_content:
        return {'upregulated': [], 'downregulated': []}
    
    result = {'upregulated': [], 'downregulated': []}
    
    # Find upregulated genes section
    up_match = re.search(r'UPREGULATED_GENES:\s*(.*?)(?=DOWNREGULATED_GENES:|$)', 
                         answer_content, re.DOTALL | re.IGNORECASE)
    
    # Find downregulated genes section
    down_match = re.search(r'DOWNREGULATED_GENES:\s*(.*?)$', 
                           answer_content, re.DOTALL | re.IGNORECASE)
    
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
    
    return result

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


def format_reward_fn(completion: str, **kwargs) -> float:
    """
    Calculates a format-based reward for a single completion.
    Rewards 1.0 if the answer tag is present and follows the expected format.
    """
    reward = 0.0

    if isinstance(completion, str):
        answer_content = extract_content(completion, "answer")
        
        if answer_content:
            # Check if the content follows the expected format
            has_upregulated = re.search(r'UPREGULATED_GENES:', answer_content, re.IGNORECASE) is not None
            has_downregulated = re.search(r'DOWNREGULATED_GENES:', answer_content, re.IGNORECASE) is not None
            
            # Check for bullet points or "NONE"
            valid_entries = re.search(r'(-\s*[^\n]+|NONE)', answer_content, re.IGNORECASE) is not None
            
            if has_upregulated and has_downregulated and valid_entries:
                reward = 1.0
            else:
                # Partial reward if some formatting is correct
                reward = 0.0
    else:
        warnings.warn(f"format_reward_fn: Expected a string for completion, but got: {type(completion)}")
    
    return reward



def accuracy_reward_fn(completion: str,
                      raw_solution_lists: dict,  # Direct access to the solution dictionary
                      **kwargs) -> float:
    """
    Calculates reward based on F1 scores between generated gene lists and solution.
    - Calculates separate F1 scores for upregulated and downregulated genes
    - Returns the average of the two F1 scores as the reward
    - Uses sklearn's f1_score for reliable calculation
    """
    
    # Default penalty for missing tags or parsing errors
    reward = -1.0

    if not isinstance(completion, str):
        logging.warning(f"AccuracyRewardFn: Expected a string for completion, got {type(completion)}.")
        return reward

    if not isinstance(raw_solution_lists, dict):
        logging.warning(f"AccuracyRewardFn: Expected a dict for raw_solution_lists, got {type(raw_solution_lists)}.")
        return reward

    try:
        # The ground truth is directly available
        ground_truth = raw_solution_lists

        # Extract and parse the generated answer
        extracted_answer = extract_content(completion, "answer")
        if extracted_answer is None:
            return -1.0  # Missing answer tag
        
        generated = parse_gene_lists(extracted_answer)
        
        # Create a universe of all genes (both up and down regulated, from both ground truth and predicted)
        all_genes = set()
        all_genes.update(ground_truth['upregulated'], ground_truth['downregulated'],
                         generated['upregulated'], generated['downregulated'])
        
        # For F1 calculation with sklearn, we need to convert to binary indicators
        # For upregulated genes
        y_true_up = np.array([1 if gene in ground_truth['upregulated'] else 0 for gene in all_genes])
        y_pred_up = np.array([1 if gene in generated['upregulated'] else 0 for gene in all_genes])
        
        # For downregulated genes
        y_true_down = np.array([1 if gene in ground_truth['downregulated'] else 0 for gene in all_genes])
        y_pred_down = np.array([1 if gene in generated['downregulated'] else 0 for gene in all_genes])
        
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
        
        # Average F1 score as the reward
        reward = (up_f1 + down_f1) / 2
                
    except Exception as e:
        logging.warning(f"AccuracyRewardFn: Error processing completion: '{completion[:100]}...'. Error: {e}")
        reward = -1.0  # Penalize parsing errors
    
    return reward



def simple_reasoning_reward_fn(completion: str, **kwargs) -> float:
    """
    Provides a small reward if the <think> tag exists and its content is not empty.
    """
    reward = 0.0
    if isinstance(completion, str):
        think_content = extract_content(completion, "think")
        if think_content: # True if not None and not an empty string
             reward = 0.1 # Small bonus for non-empty reasoning
    else:
        warnings.warn(f"simple_reasoning_reward_fn: Expected a string for completion, got {type(completion)}")
        # reward remains 0.0 for non-string inputs

    return reward


def overall_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    Combines multiple reward functions to calculate an overall reward.
    """
    try:
        # Extract raw_solution_lists from the ground_truth
        # make sure that the ground_truth items are already lists
        processed_ground_truth = {}
        for key in ['downregulated', 'upregulated']:
            if key in ground_truth:
                value = ground_truth[key]
                # Convert to list if it's a numpy array, otherwise keep as is
                if hasattr(value, 'tolist'):
                    processed_ground_truth[key] = value.tolist()
                else:
                    processed_ground_truth[key] = value
        
        # Call individual reward components with appropriate parameters
        accuracy_val = accuracy_reward_fn(
            completion=solution_str,
            raw_solution_lists=processed_ground_truth
        )
        
        format_val = format_reward_fn(
            completion=solution_str
        )
        
        reasoning_val = simple_reasoning_reward_fn(
            completion=solution_str
        )
        
        # Combine the rewards (weights can be adjusted as needed)
        combined_reward = (
            0.5 * accuracy_val +
            0.3 * format_val +
            0.2 * reasoning_val
        )
        
        return float(combined_reward)
        
    except Exception as e:
        logging.warning(f"Error in overall_reward_fn: {e}")
        # Default to a penalty in case of errors
        return -1.0