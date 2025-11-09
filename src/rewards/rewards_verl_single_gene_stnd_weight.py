import torch
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
                       solution_item: str, # The ground truth answer string for this completion
                       **kwargs) -> float:
    """
    Calculates reward based on whether the generated answer in a single completion
    matches the single solution_item.
    - Rewards 1.0 for a correct answer within <answer> tags.
    - Rewards -0.5 if <answer> tag is present but content is incorrect.
    - Rewards -1.0 if <answer> tag is missing or other parsing errors occur.
    """
    # Default penalty for missing tags, parsing errors, or other issues.
    # This ensures a value is always returned.
    reward = -1.0

    if not isinstance(completion, str):
        logging.warning(f"AccuracyRewardFn: Expected a string for completion, got {type(completion)}.")
        return reward 

    if not isinstance(solution_item, str):
        logging.warning(f"AccuracyRewardFn: Expected a string for solution_item, got {type(solution_item)}. Cannot score accuracy.")
        # Depending on policy, could return a specific penalty or handle differently.
        return reward # Or perhaps 0.0 if missing solution means neutral score.

    # Normalize the ground truth solution
    ground_truth_answer = solution_item.strip().lower()

    try:
        extracted_answer = extract_content(completion, "answer") # Use existing robust extraction
        
        if extracted_answer is not None: # extract_content returns None if tag missing or empty
            generated_answer_normalized = extracted_answer.strip().lower()
            if generated_answer_normalized == ground_truth_answer:
                # Optional: print(f"AccuracyRewardFn: Correct answer found: '{generated_answer_normalized}'")
                reward = 1.0  # Correct answer
            else:
                reward = -0.5 # Answer tag present, but content incorrect
        # else: reward remains -1.0 (if <answer> tag is missing or content is empty)
            
    except Exception as e:
        # This catch is general. extract_content should handle regex errors.
        # This might catch issues if extract_content had an unexpected failure.
        logging.warning(f"AccuracyRewardFn: Error processing completion: '{completion[:100]}...'. Error: {e}")
        reward = -1.0 # Penalize parsing errors (already the default)

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