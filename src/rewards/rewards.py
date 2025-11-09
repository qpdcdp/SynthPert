import re
import torch
import warnings
import os
from accelerate import PartialState
import logging

# Ensure PartialState can be initialized locally if not distributed
try:
    _ = PartialState()
except Exception as e:
    if "torch.distributed.distributed_c10d" in str(e):
         # This error happens if torch.distributed isn't initialized
         # We can define a dummy state for local runs if needed,
         # but using PartialState as is works correctly in accelerate environments.
         # Let's trust accelerate handles this.
         pass # Keep the PartialState import and usage as is

# Modified extract_content function (Remove Debug Prints)
def extract_content(text, tag):
    """Extracts content between the first occurrence of <tag>...</tag>."""
    # Ensure text is a string and not empty
    if not isinstance(text, str) or not text:
        return None
        
    # Check for existence of both start and end tags before searching
    start_tag = f"<{tag}"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
         return None # Speed up by failing early if tags are missing

    # Updated pattern to be more robust and greedy within the tags
    # .*? matches any character non-greedily
    # [^>]* allows attributes in the opening tag
    pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    # The match object check is crucial
    return match.group(1).strip() if match else None


# Modified format_reward_fn
def format_reward_fn(prompts: list[str], # prompts is also passed by GRPOTrainer
                       completions: list[str], # Corrected: should be list[str]
                       **kwargs) -> list[float]:
    rewards = []
    valid_answers = {"upregulated", "downregulated", "not differentially expressed"}

    # Optional: Print the type and first few elements of completions ONCE per call on rank 0
    # This helps verify the structure GRPOTrainer is actually passing.
    # state_debug = PartialState()
    # if state_debug.is_main_process:
    #     if completions: # Check if not empty
    #         print(f"[format_reward_fn DEBUG on Rank 0] Type of completions: {type(completions)}")
    #         print(f"[format_reward_fn DEBUG on Rank 0] First completion type: {type(completions[0]) if completions else 'N/A'}")
    #         print(f"[format_reward_fn DEBUG on Rank 0] First completion content (first 100 chars): '{str(completions[0])[:100] if completions else 'N/A'}'")
    #     else:
    #         print("[format_reward_fn DEBUG on Rank 0] Completions list is empty.")


    for full_text in completions: # Iterate directly over the strings
        reward = 0.0
        if isinstance(full_text, str): # Ensure it's a string
            think_content = extract_content(full_text, "think")
            answer_content = extract_content(full_text, "answer")

            if think_content and answer_content:
                 if answer_content.lower() in valid_answers:
                      reward = 1.0
                 else:
                      reward = 0.0 # Or some other value for format OK, content wrong
            # else: reward remains 0.0
        else:
            # Handle case where an element in completions is not a string (shouldn't happen with GRPOTrainer)
            warnings.warn(f"format_reward_fn: Encountered non-string element in completions: {type(full_text)}")
        rewards.append(reward)

    # # Your existing logging block for sample results (this is good)
    # state_log = PartialState()
    # if state_log.is_main_process and len(completions) > 0:
    #     print("\n--- Format Reward Debug ---")
    #     for i in range(min(3, len(completions))):
    #         text_to_log = completions[i] if isinstance(completions[i], str) else "<<InvalidCompletionType>>"
    #         think_log = extract_content(text_to_log, "think") # This will call your print inside extract_content
    #         answer_log = extract_content(text_to_log, "answer") # This will also call your print
    #         print(f"Sample {i}: Text: '{text_to_log[:100]}...'")
    #         print(f"  <think> extracted: '{think_log}'")
    #         print(f"  <answer> extracted: '{answer_log}'")
    #         print(f"  Valid Answer: {answer_log.lower() in valid_answers if answer_log else False}")
    #         print(f"  Reward: {rewards[i] if i < len(rewards) else 'N/A'}") # Safety for rewards list
    #     print("---------------------------\n")

    return rewards



def accuracy_reward_fn(prompts: list[str],
                       completions: list[str],
                       solution: list[str], # This comes from your dataset's "solution" field
                       **kwargs):
    """
    Calculates reward based on whether the generated answer matches the solution.
    It infers the number of generations per prompt based on the lengths of
    'prompts' and 'completions' lists.

    Args:
        prompts (list[str]): List of prompts.
        completions (list[str]): List of generated completions.
                                 Expected length: len(prompts) * actual_num_generations.
        solution (list[str]): List of correct solutions (ground truth answers),
                              aligned with `prompts`. Expected length: len(prompts).
        **kwargs: Additional keyword arguments (e.g., completions_ids, other dataset columns).
    """
    rewards = []

    if not prompts:
        return [] # No prompts, no rewards

    if len(prompts) == 0: # Should be caught by 'if not prompts' but good for clarity
        return []

    # Infer the number of generations per prompt
    # This is the *actual* number of completions generated for each prompt in this batch
    if len(completions) % len(prompts) != 0:
        warnings.warn(
            f"AccuracyRewardFn: Number of completions ({len(completions)}) "
            f"is not an even multiple of the number of prompts ({len(prompts)}). "
            f"Reward calculation might be incorrect. Assuming 1 generation if completions exist."
        )
        # This state is problematic. How to best handle?
        # If len(completions) > 0, we could assume num_actual_generations = 1 and process only first len(prompts) completions.
        # Or return default penalties. For now, let's try to proceed cautiously.
        if len(completions) == 0:
            return [0.0] * (len(prompts) * 1) # Or whatever default for no completions
        num_actual_generations = len(completions) // len(prompts) if len(prompts) > 0 else 1
        if num_actual_generations == 0 and len(completions) > 0: # if len(prompts) > len(completions)
            num_actual_generations = 1


    else:
        num_actual_generations = len(completions) // len(prompts)

    if num_actual_generations == 0 and len(completions) > 0: # e.g. 3 prompts, 2 completions
         warnings.warn(f"AccuracyRewardFn: Fewer completions ({len(completions)}) than prompts ({len(prompts)}). Cannot reliably assign rewards.")
         return [-1.0] * len(completions) # Penalize all available completions


    if len(solution) != len(prompts):
        warnings.warn(
            f"AccuracyRewardFn: Mismatch in lengths: {len(prompts)} prompts, but {len(solution)} solutions. "
            "Cannot reliably calculate accuracy. Rewards might be misaligned."
        )
        # Return a list of rewards matching the number of completions, with a default penalty
        return [-1.0] * len(completions)

    for i in range(len(prompts)):
        prompt_solution_text = solution[i].strip().lower()

        for gen_idx in range(num_actual_generations):
            completion_flat_idx = i * num_actual_generations + gen_idx

            # Safety check, though theoretically, the loop structure should prevent this
            # if num_actual_generations was inferred correctly.
            if completion_flat_idx >= len(completions):
                logging.error(
                    f"AccuracyRewardFn: Index out of bounds for completions. "
                    f"Prompt_idx={i}, Gen_idx={gen_idx}, Total_gens_per_prompt={num_actual_generations}, "
                    f"Calculated_completion_idx={completion_flat_idx}, Len_completions={len(completions)}"
                )
                rewards.append(-1.0) # Error condition
                continue

            current_completion = completions[completion_flat_idx]

            try:
                # More robust regex to handle potential newlines/spaces within tags
                match = re.search(r"<answer>(.*?)</answer>", current_completion, re.DOTALL | re.IGNORECASE)
                if match:
                    generated_answer = match.group(1).strip().lower()
                    if generated_answer == prompt_solution_text:
                        print(f"AccuracyRewardFn: Correct answer found: '{generated_answer}'")
                        rewards.append(1.0)  # Correct answer
                    else:
                        rewards.append(-0.5) # Answer tag present, but incorrect content
                else:
                    # No <answer> tag found
                    rewards.append(-1.0) # Format error, penalize
            except Exception as e:
                logging.warning(f"AccuracyRewardFn: Error parsing completion: '{current_completion}'. Error: {e}")
                rewards.append(-1.0) # Penalize parsing errors

    if len(rewards) != len(completions):
        # This would indicate a logic error in the loops above
        warnings.warn(
            f"AccuracyRewardFn: Mismatch between number of calculated rewards ({len(rewards)}) "
            f"and number of completions ({len(completions)}). Filling with defaults."
        )
        # Fallback: return a list of default rewards matching completions length
        return [-1.0] * len(completions)


    return rewards

# Keep simple_reasoning_reward_fn as is, but note potential redundancy with new format_reward_fn
# If format_reward_fn gives 1.0 for having tags, this one giving 0.1 for <think> might be small.
# You might remove this function or adjust its reward/logic based on your desired training signal.
def simple_reasoning_reward_fn(completions: list[list[str]], **kwargs) -> list[float]: # Changed return type hint
    """
    Provides a small reward if the <think> tag exists and is not empty.
    """
    rewards = []
    for text_list in completions:
        reward = 0.0
        if text_list and isinstance(text_list[0], str):
            full_text = text_list[0]
            think_content = extract_content(full_text, "think")
            if think_content: # Check if not None and not empty string
                 reward = 0.1 # Small bonus for non-empty reasoning

        rewards.append(reward)
    return rewards # Return the Python list


def overall_reward_fn(prompts: list[str],
                       completions: list[str],
                       solution: list[str], # This comes from your dataset's "solution" field
                       **kwargs):
    """
    Combines multiple reward functions to calculate an overall reward.
    """
    rewards = []
    # Call the other reward functions
    accuracy_rewards = accuracy_reward_fn(prompts, completions, solution, **kwargs)
    format_rewards = format_reward_fn(prompts, completions, **kwargs)
    reasoning_rewards = simple_reasoning_reward_fn(completions, **kwargs)

    for i in range(len(prompts)):
        # Combine the rewards (you can adjust the weights as needed)
        combined_reward = (
            0.5 * accuracy_rewards[i] +
            0.3 * format_rewards[i] +
            0.2 * reasoning_rewards[i]
        )
        rewards.append(combined_reward)

    return rewards