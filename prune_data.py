import pandas as pd
import io
import numpy as np

# --- 1. Setup: Create a sample DataFrame ---
# I've added more data to your example to properly demonstrate the pruning logic.
# - 'k562' cell_type will have too many 'not differentially expressed' rows and will be pruned.
# - 'hct116' cell_type will be within the threshold and will be left untouched.



# In a real scenario, you would load your file like this:
# df = pd.read_csv('your_file.csv')
df = pd.read_csv("/novo/projects/departments/mi/lwph/CellPert/output/synth_data/single_gene_prediction/o4_mini/with_critic/rpe1_split/default_generator_prompt_critic_default_prompt_critic_threshold_excellent_only.csv")

print("--- Original DataFrame ---")
print(df)
print("\n")


# --- 2. Define constants and prepare for pruning ---
ANSWER_TO_PRUNE = "not differentially expressed"
THRESHOLD = 1/3

# This list will store the indices of the rows we decide to drop.
indices_to_drop = []

print("--- Analyzing and Calculating Rows to Prune ---")
# Group the DataFrame by the 'cell_type' column
for cell_type_name, group in df.groupby('cell_type'):
    
    total_rows = len(group)
    
    # Isolate the rows that match the answer we want to prune
    nde_rows = group[group['true_answer'] == ANSWER_TO_PRUNE]
    nde_count = len(nde_rows)
    
    # Calculate the maximum number of 'nde' rows allowed for this group
    max_allowed_nde = int(np.floor(total_rows * THRESHOLD))
    
    print(f"\nAnalyzing cell_type: '{cell_type_name}'")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with '{ANSWER_TO_PRUNE}': {nde_count}")
    print(f"  Allowed max (1/3 of total): {max_allowed_nde}")
    
    # Check if the number of 'nde' rows exceeds the threshold
    if nde_count > max_allowed_nde:
        num_to_remove = nde_count - max_allowed_nde
        print(f"  -> Pruning needed. Will remove {num_to_remove} rows.")
        
        # Randomly sample the 'nde' rows to identify which ones to drop
        # Using random_state makes the random selection reproducible
        rows_to_remove = nde_rows.sample(n=num_to_remove, random_state=42)
        
        # Add the indices of these rows to our master drop list
        indices_to_drop.extend(rows_to_remove.index)
        
    else:
        print("  -> No pruning needed for this group.")

# --- 3. Prune the DataFrame ---
pruned_df = df.drop(indices_to_drop)

print("\n\n--- Pruning Complete ---")
print(f"Total rows dropped: {len(indices_to_drop)}")


# --- 4. Verification ---
print("\n--- Verification of Pruned DataFrame ---")
for cell_type_name, group in pruned_df.groupby('cell_type'):
    total_rows_after = len(group)
    nde_count_after = len(group[group['true_answer'] == ANSWER_TO_PRUNE])
    
    print(f"\nVerifying cell_type: '{cell_type_name}'")
    print(f"  Total rows after pruning: {total_rows_after}")
    print(f"  Rows with '{ANSWER_TO_PRUNE}' after pruning: {nde_count_after}")
    
    # The condition should now be met for all groups
    is_valid = nde_count_after <= int(np.floor(total_rows_after * THRESHOLD))
    print(f"  Is ratio valid (<= 1/3)? -> {is_valid}")


# --- 5. Display and Save the Result ---
print("\n\n--- Final Pruned DataFrame ---")
print(pruned_df)

# To save the final result to a new CSV file:
pruned_df.to_csv('/novo/projects/departments/mi/lwph/CellPert/output/synth_data/single_gene_prediction/o4_mini/with_critic/rpe1_split/default_generator_prompt_critic_default_prompt_critic_threshold_excellent_only_PRUNED.csv', index=False)
print("\nPruned data saved to 'pruned_data.csv'")