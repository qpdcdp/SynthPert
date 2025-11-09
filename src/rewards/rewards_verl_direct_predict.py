import torch
from sklearn.metrics import f1_score
import numpy as np
import warnings
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


def format_reward_fn(completion: str, **kwargs) -> float:
    """
    Calculates a format-based reward for a single completion.
    Rewards 1.0 if both <think> and <answer> tags are present and the answer
    is one of the valid predefined answers.
    """
    reward = 0.0
    # This set can be defined outside if it's static and shared
    valid_answers = {"upregulated", "downregulated", "not differentially expressed"}

    if isinstance(completion, str):
        think_content = extract_content(completion, "think")
        answer_content = extract_content(completion, "answer")

        if think_content and answer_content: # Both tags must be present and non-empty
             if answer_content.lower() in valid_answers:
                  reward = 1.0
             else:
                  # Tags are present, but answer content is not in the valid set.
                  # Could assign a partial reward or specific penalty here if desired.
                  reward = 0.0 # Currently, 0.0 if answer content is invalid
        # If either tag is missing or empty, reward remains 0.0
    else:
        warnings.warn(f"format_reward_fn: Expected a string for completion, but got: {type(completion)}")
        # reward remains 0.0 for non-string inputs

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
            up_f1 = f1_score(y_true_up, y_pred_up)
        
        # For downregulated genes
        if np.sum(y_true_down) == 0 and np.sum(y_pred_down) == 0:
            down_f1 = 1.0  # Perfect agreement on empty lists
        elif np.sum(y_true_down) == 0 or np.sum(y_pred_down) == 0:
            down_f1 = 0.0  # One list is empty, the other isn't
        else:
            down_f1 = f1_score(y_true_down, y_pred_down)
        
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
    Adapted to veRL's expected parameter structure.
    
    Args:
        data_source (str): Dataset source identifier.
        solution_str (str): The model-generated response.
        ground_truth (str): The correct answer.
        extra_info (dict, optional): Additional information (not used).
        
    Returns:
        float: The combined reward score
    """
    try:
        # Call individual reward components with appropriate parameters
        accuracy_val = accuracy_reward_fn(
            completion=solution_str,
            solution_item=ground_truth
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