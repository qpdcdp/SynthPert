import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np # Import numpy for efficient label calculation
import warnings # To warn about inconsistencies

from src.utils.enrichr_old import find_pathways, generate_prompt
from src.utils.llm_context import get_llm_context

class DiffExpressionDatasetTestEnrichr(Dataset):
    def __init__(self, csv_file, split="train", prompt_mode: str = "default", context: str = "enrichr"):
        """
        Dataset for differential gene expression across multiple cell types,
        loading data from paired differential expression (-de.csv) and
        direction (-dir.csv) files.

        Args:
            csv_file: Path to the consolidated CSV file.
            split: Data split to use ("train", "test", or None for all).
                   Filters based on a 'split' column if present in the CSV file.
            prompt_mode: (Currently unused in snippet, but kept for consistency)
            context: Specifies the type of context to add to the prompt ("enrichr", "llm", or None).
        """
        self.data = []
        self.csv_file = Path(csv_file)
        self.split = split
        self.prompt_mode = prompt_mode
        self.context = context
        self.cell_type_col = "jurkat" # Store the cell type column name - Assuming this is constant or needs adjustment if dynamic
        self.pert_to_genes = {} # Initialize dictionary to store genes per perturbation

        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        print(f"Loading data from: {self.csv_file}")
        df = pd.read_csv(self.csv_file)

        # --- Input Validation ---
        required_columns = ['pert', 'gene', 'label']
        if self.split is not None:
            required_columns.append('split')

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {self.csv_file}: {', '.join(missing_cols)}")

        # --- Filtering by Split ---
        if self.split is not None:
            if 'split' not in df.columns:
                 warnings.warn(f"'split' column not found in {self.csv_file}, but split='{self.split}' was requested. Loading all data.")
            else:
                original_len = len(df)
                df = df[df['split'] == self.split].copy()
                print(f"Filtered data for split '{self.split}'. Kept {len(df)} out of {original_len} rows.")
                if len(df) == 0:
                     warnings.warn(f"No data found for split '{self.split}' in {self.csv_file}. Dataset will be empty.")
                     # If df is empty, return early or handle appropriately
                     # For now, we'll let it proceed, but the dataset will be empty.

        # --- Data Processing ---
        if not df.empty: # Proceed only if DataFrame is not empty after filtering
            # Check label values
            valid_labels = {0, 1}
            if not df['label'].isin(valid_labels).all():
                 invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
                 warnings.warn(f"Found unexpected values in 'label' column: {invalid_labels}. Expected 0 or 1. These rows might cause issues.")
                 # Optionally filter out invalid labels:
                 # df = df[df['label'].isin(valid_labels)].copy()
                 # print(f"Removed rows with invalid labels. Kept {len(df)} rows.")

            # --- Pre-calculate genes per perturbation for Enrichr ---
            # Create this map *before* converting df to list of dicts for efficiency
            if self.context == "enrichr":
                print("Pre-calculating gene lists per perturbation for Enrichr context...")
                # Group by 'pert' and collect all 'gene' values into a list
                self.pert_to_genes = df.groupby('pert')['gene'].apply(list).to_dict()
                print(f"Found gene lists for {len(self.pert_to_genes)} unique perturbations.")
                # Simple check if map is populated
                if not self.pert_to_genes:
                     warnings.warn("Perturbation to gene mapping is empty. Enrichr context might not work as expected.")


            # Convert DataFrame rows to list of dictionaries
            for _, row in df.iterrows():
                try:
                    label_val = int(row['label'])
                    if label_val not in valid_labels:
                        # Skip rows with invalid labels if not filtered above
                        continue


                    self.data.append({
                        "pert": str(row['pert']),
                        "gene": str(row['gene']),
                        "label": label_val, # Ensure label is integer 0 or 1
                    })
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Skipping row due to conversion error (label='{row['label']}'): {e}\nRow data: {row.to_dict()}")
                except KeyError as e:
                     warnings.warn(f"Skipping row due to missing key '{e}': {row.to_dict()}") # Should be less likely now with .get


        # Final print statement
        print(f"\nInitialized dataset for split '{self.split or 'all'}'. Total samples loaded: {len(self.data)}")

        self.label_map = {0: "down", 1: "up"}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_solution = self.label_map[item["label"]]

        system_prompt_string = (
            "You are an expert bioinformatician analyzing gene regulation upon CRISPRi knockdown. "
            "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
            "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
            "gene interactions, and cell-specific context. "
            "Then, provide the final answer ('up', or 'down') within <answer> </answer> tags. "
            "Example: <think> [Your reasoning here] </think><answer> [up/down] </answer>"
        )

        user_prompt_string = "" # Initialize

        # User prompt generation based on context
        if self.context == "enrichr":
            current_pert = item["pert"]
            # Retrieve the pre-calculated list of all genes for this perturbation
            all_genes_for_pert = self.pert_to_genes[current_pert] # Use .get for safety

            print(f"DEBUG: Getting pathways for pert {current_pert} with {len(all_genes_for_pert)} genes...") # ADD THIS
            sys.stdout.flush()
            # Ensure find_pathways can handle a list of genes
            pathways = find_pathways(all_genes_for_pert)
            print(all_genes_for_pert)
            print(f"Pathways found for pert '{current_pert}': {pathways}")
            sys.stdout.flush()
            prompt_context = generate_prompt(pathways, all_genes_for_pert)

            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell jurkat cell line using CRISPR interference. It is either up or down regulated. Determine the direction of regulation. "
                f"I am providing gene enrichment analysis data from Enrichr. The input gene list for this analysis includea all up and down regulated genes after knocking down the {item['pert']} gene. " # Clarified source
                f"This information may inform your prediction:\n{prompt_context}"
            )
        elif self.context=="llm":
            # Get the context from the LLM
            model_name,llm_context = get_llm_context(item["pert"], item["gene"], item["cell_type"])

            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
                f"I am providing gene enrichment analysis from {model_name} related to the {item['gene']} gene and the "
                f"knocked down {item['pert']} gene, which may inform your prediction:\n{llm_context}"
            )
        else:
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
            )

        # Create the chat-formatted prompt
        formatted_prompt = [
            {"role": "system", "content": system_prompt_string},
            {"role": "user", "content": user_prompt_string}
        ]

        return {
            "pert": item["pert"],
            "gene": item["gene"],
            "label": item["label"],        # Numerical label (0, 1, 2) - potentially useful metadata
            "cell_type": "rpe1", # Assuming this is constant or needs adjustment if dynamic
            "prompt": formatted_prompt,   # List of dicts for chat format
            "solution": text_solution     # Text label ("no", "down", "up") - for accuracy reward
        }