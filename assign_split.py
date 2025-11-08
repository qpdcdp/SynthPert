import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

def create_stratified_split(df):
    """
    Adds a 'split' column to the DataFrame with an 80/20 train/test split.

    The split is stratified by the combination of 'insulin_resistance' and
    'insulin_stimulation' columns, and it ensures that unique values from the
    'pert' column are held out for the test set.

    Args:
        df (pd.DataFrame): The input DataFrame with columns 'pert',
                           'insulin_resistance', and 'insulin_stimulation'.

    Returns:
        pd.DataFrame: The DataFrame with the new 'split' column.
    """
    # Create an empty 'split' column
    df['split'] = ''

    # Group the DataFrame by the two conditions
    grouped = df.groupby(['insulin_resistance', 'insulin_stimulation'])

    # A set to hold all the 'pert' values assigned to the test set
    test_perts = set()

    for _, group in grouped:
        # Get the unique 'pert' values for the current group
        unique_perts = group['pert'].unique()

        # Split the unique 'pert' values into training and testing sets
        train_perts_group, test_perts_group = train_test_split(
            unique_perts, test_size=0.20, random_state=42
        )

        # Add the test 'pert' values from this group to the overall set
        test_perts.update(test_perts_group)

    # Assign 'train' or 'test' to the 'split' column based on the 'pert' value
    df['split'] = df['pert'].apply(lambda x: 'test' if x in test_perts else 'train')

    return df


df = pd.read_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt.csv")


# Create the stratified split
df_split = create_stratified_split(df)

# Display the DataFrame with the new 'split' column
print(df_split)
df_split.to_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt_pert_split.csv", index=False)
# Verify the split ratio for each group
print("\nVerification of the split:")
for name, group in df_split.groupby(['insulin_resistance', 'insulin_stimulation']):
    print(f"\nCondition: {name}")
    print(group['split'].value_counts(normalize=True))

    import pandas as pd
from io import StringIO

def create_split_on_stimulation(df):
    """
    Adds a 'split' column to the DataFrame based on the 'insulin_stimulation'
    axis.

    Assigns 'train' to rows where 'insulin_stimulation' is 0, and 'test' to
    rows where 'insulin_stimulation' is 1.

    Args:
        df (pd.DataFrame): The input DataFrame with the 'insulin_stimulation'
                           column.

    Returns:
        pd.DataFrame: The DataFrame with the new 'split' column.
    """
    # Create the 'split' column by mapping the values from 'insulin_stimulation'
    # 0 maps to 'train', 1 maps to 'test'
    df['split'] = df['insulin_stimulation'].apply(lambda x: 'test' if x == 0 else 'train')
    return df


# Read the data into a pandas DataFrame
# In a real scenario, you would use: df = pd.read_csv('your_file.csv')
df = pd.read_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt.csv")


# Create the split based on the insulin_stimulation axis
df_split = create_split_on_stimulation(df)

# Display the DataFrame with the new 'split' column
print(df_split)
df_split.to_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt_pert_split_stimulation_0.csv", index=False)
# Verify the split by showing the direct mapping
print("\nVerification of the split:")
# This table shows how 'insulin_stimulation' values were mapped to the 'split' column
print(pd.crosstab(df_split['insulin_stimulation'], df_split['split']))


import pandas as pd
from io import StringIO

def create_split_on_resistance(df):
    """
    Adds a 'split' column to the DataFrame based on the 'insulin_resistance'
    axis.

    Assigns 'train' to rows where 'insulin_resistance' is 0, and 'test' to
    rows where 'insulin_resistance' is 1.

    Args:
        df (pd.DataFrame): The input DataFrame with the 'insulin_resistance'
                           column.

    Returns:
        pd.DataFrame: The DataFrame with the new 'split' column.
    """
    # Create the 'split' column by mapping the values from 'insulin_resistance'
    # 0 maps to 'train', 1 maps to 'test'
    df['split'] = df['insulin_resistance'].apply(lambda x: 'test' if x == 1 else 'train')
    return df


df = pd.read_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt.csv")

# Create the split based on the insulin_resistance axis
df_split = create_split_on_resistance(df)

# Display the DataFrame with the new 'split' column
print(df_split)
df_split.to_csv("/novo/projects/departments/mi/lwph/CellPert/data_metabolic_flux/df_Flux-melt_pert_split_resistance_1.csv", index=False)
# Verify the split by showing the direct mapping
print("\nVerification of the split:")
# This table shows how 'insulin_resistance' values were mapped to the 'split' column
print(pd.crosstab(df_split['insulin_resistance'], df_split['split']))