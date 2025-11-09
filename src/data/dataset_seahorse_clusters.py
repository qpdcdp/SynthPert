import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import warnings
import logging

# Assuming these utilities are in your project structure
from sklearn.model_selection import train_test_split
from src.utils.enrichr_old import find_pathways, generate_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# TODO: add pathway prediction functionality to grab from other folder
# TODO: add mechanism for checking performance on prediction of genes expressed to synth csv

class SeahorseClusterDataset(Dataset):
    def __init__(self, csv_path, split, tokenizer=None,
                 prompt_mode: str = "default",
                 train_mode: str = "SFT",
                 context: str = "none",
                 tool: str = "none",
                 exclude_sft_csv=None,
                 eval_unique_genes_to_test_only=False,
                 generate_all_non_de_samples=False,
                 generate_4x_non_de_samples=False,
                 **kwargs):
        """
        Initializes the dataset from a single CSV file containing Seahorse cluster explanations and a split column.

        Args:
            csv_path (str): Path to the single input CSV file.
            split (str or None): The dataset split to load ('train', 'test', or None for all).
            exclude_sft_csv (str, optional): Path to a CSV file containing samples to exclude.
            eval_unique_genes_to_test_only (bool): If True, ensures test set perturbations ('pert') are not in the train set.
            **kwargs: Catches other unused arguments.
        """
        # --- Configuration ---
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.split = split
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.context = context
        self.tool = tool

        if self.context not in ["none", "default"]:
            warnings.warn(f"Context mode '{self.context}' is not suitable for this dataset format and will be ignored.")
            self.context = "none"

        # --- Data Loading and Processing ---
        # 1. Get train perturbations if needed for test set uniqueness
        self.eval_unique_genes_to_test_only = eval_unique_genes_to_test_only
        self.train_perts = set()
        if self.split == 'test' and self.eval_unique_genes_to_test_only:
            logging.info("Scanning for training set perturbations to ensure test set uniqueness...")
            self.train_perts = self._get_train_perts()
            logging.info(f"Found {len(self.train_perts)} unique perturbations in the training data. "
                         f"These will be excluded from the test set.")

        # 2. Load and process the data file using the split column
        all_data_df = self._load_and_process_data()


        # 4. Finalization
        if all_data_df.empty:
            raise RuntimeError(
                f"Dataset is empty. No data loaded for split '{self.split or 'all'}' from {self.csv_path} "
                f"with current settings."
            )

        self.data = all_data_df.to_dict('records')

        logging.info(
            f"\n--- Dataset Initialized ---\n"
            f"Split: '{self.split or 'all'}'\n"
            f"Total samples: {len(self.data)}\n"
            f"Detected Labels: {list(self.label_map.values())}\n"
            f"---------------------------\n"
        )

    def _get_train_perts(self):
        """
        Scans the single data file to get a set of unique perturbations present in the 'train' split.
        """
        if not self.csv_path.is_file():
            warnings.warn(f"Cannot get train perturbations. File not found: {self.csv_path}")
            return set()
        try:
            df_full = pd.read_csv(self.csv_path, usecols=['pert', 'split'])
        except Exception as e:
            warnings.warn(f"Could not read {self.csv_path.name} to get train perturbations: {e}.")
            return set()

        if 'split' not in df_full.columns:
            warnings.warn("Cannot determine train split for unique perturbation filtering. No 'split' column found.")
            return set()

        return set(df_full[df_full['split'] == 'train']['pert'].unique())

    def _load_and_process_data(self):
        """
        Loads data from the CSV, creates dynamic labels, and handles splitting based on the 'split' column.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Input file not found at: {self.csv_path}")

        try:
            # Read all necessary columns including 'split'
            df = pd.read_csv(self.csv_path, usecols=['pert', 'Seahorse_Cluster_Explanation', 'split'])
        except Exception as e:
            raise IOError(f"Error reading the CSV file at {self.csv_path}: {e}")

        # 1. Validate and Clean
        required_cols = {'pert', 'Seahorse_Cluster_Explanation', 'split'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input CSV must contain the columns: {required_cols}")
        df.dropna(subset=list(required_cols), inplace=True)

        # 2. Dynamic Label Creation
        unique_explanations = sorted(df['Seahorse_Cluster_Explanation'].unique().tolist())
        string_to_int_map = {}
    
        
        for i, explanation in enumerate(unique_explanations):
            string_to_int_map[explanation] = i + len(string_to_int_map)

        self.label_map = {v: k for k, v in string_to_int_map.items()}
        df['label'] = df['Seahorse_Cluster_Explanation']

        # 3. Add Missing Contextual Columns
        df['gene'] = df['Seahorse_Cluster_Explanation']
        df['cell_type'] = 'primary human adipocytes'

        # 4. Apply Train/Test Split using the 'split' column
        if self.split:
            df_split_filtered = df[df['split'] == self.split].copy()
        else: # If no split is specified, use all data
            df_split_filtered = df.copy()

        if df_split_filtered.empty:
            warnings.warn(f"No data found for split '{self.split}' in the CSV file.")
            return pd.DataFrame()

        # 5. Filter Test Set for Unique Perturbations (if enabled)
        if self.split == 'test' and self.eval_unique_genes_to_test_only and self.train_perts:
            initial_count = len(df_split_filtered)
            df_split_filtered = df_split_filtered[~df_split_filtered['pert'].isin(self.train_perts)]
            if (filtered_count := initial_count - len(df_split_filtered)) > 0:
                logging.info(f"Excluded {filtered_count} test samples with perturbations also found in the train set.")
        
        return df_split_filtered[['pert', 'gene', 'label', 'cell_type']]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_solution = item["label"]
        
        synth_data_tool_call = ""
        tool_call_option_str_1 = ""
        tool_call_option_str_2 = ""
        if self.tool == "enrichr":
            synth_data_tool_call = ", I would like to think one more time about the pathways and gene interactions before answering"
            tool_call_option_str_1 = " or, I do not know" + synth_data_tool_call
            tool_call_option_str_2 = "/ I do not know"

        # Define system_prompt_string based on prompt_mode
        # This part is complex and has item-specific formatting needs for some modes.
        # We'll construct it dynamically within __getitem__ to ensure correctness.
        
        if self.prompt_mode == "default":
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in oxygen consumption rate and extracellular acidification rate by siRNA-mediated gene silencing in a Seahorse mitostress test. "
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, and gene functions. "
                f"Then, choose one option from the following metabolic states and place your choice within <answer> </answer> tags: 'Increased oxygen consumption rate to extracellular acidification rate ratio', 'Increased extracellular acidification rate and ATP-linked respiration', 'Increased Maximal Respiration', 'Increased Proton Leak', 'Loss of oxidative metabolism', or 'No Change'{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [Increased oxygen consumption rate to extracellular acidification rate ratio / Increased extracellular acidification rate and ATP-linked respiration / Increased Maximal Respiration / Increased Proton Leak / Loss of oxidative metabolism / No Change'{tool_call_option_str_2}] </answer>"
            )
        elif self.prompt_mode == "synth_data": # "o3_synth_data" in original, assuming typo in prompt
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in oxygen consumption rate and extracellular acidification rate by siRNA-mediated gene silencing in a Seahorse mitostress test. "
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, and gene functions. "
                f"Then, choose one option from the following metabolic states and place your choice within <answer> </answer> tags: 'Increased oxygen consumption rate to extracellular acidification rate ratio', 'Increased extracellular acidification rate and ATP-linked respiration', 'Increased Maximal Respiration', 'Increased Proton Leak', 'Loss of oxidative metabolism', or 'No Change'{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [Increased oxygen consumption rate to extracellular acidification rate ratio / Increased extracellular acidification rate and ATP-linked respiration / Increased Maximal Respiration / Increased Proton Leak / Loss of oxidative metabolism / No Change'{tool_call_option_str_2}] </answer>"
            )
        elif self.prompt_mode == "synth_data_with_ans":
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in oxygen consumption rate and extracellular acidification rate by siRNA-mediated gene silencing in a Seahorse mitostress test. "
                f"The regulatory effect of knocking down the {item['pert']} gene is given to you {text_solution}."
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, and gene functions. "
                f"Then, choose one option from the following metabolic states and place your choice within <answer> </answer> tags: 'Increased oxygen consumption rate to extracellular acidification rate ratio', 'Increased extracellular acidification rate and ATP-linked respiration', 'Increased Maximal Respiration', 'Increased Proton Leak', 'Loss of oxidative metabolism', or 'No Change'{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [Increased oxygen consumption rate to extracellular acidification rate ratio / Increased extracellular acidification rate and ATP-linked respiration / Increased Maximal Respiration / Increased Proton Leak / Loss of oxidative metabolism / No Change'{tool_call_option_str_2}] </answer>"
            )
        elif self.prompt_mode == "o3_test":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                "Consider relevant pathways "
                "(e.g., cancer biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following metabolic states and place your choice within <answer> </answer> tags: 'Increased oxygen consumption rate to extracellular acidification rate ratio', 'Increased extracellular acidification rate and ATP-linked respiration', 'Increased Maximal Respiration', 'Increased Proton Leak', 'Loss of oxidative metabolism', or 'No Change'{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [Increased oxygen consumption rate to extracellular acidification rate ratio / Increased extracellular acidification rate and ATP-linked respiration / Increased Maximal Respiration / Increased Proton Leak / Loss of oxidative metabolism / No Change'{tool_call_option_str_2}] </answer>"
            )
        user_prompt_string = (
                f"Choose the metabolic state when {item['pert']} gene is silenced by siRNAs "
                f"in a primary human adipocytes."
            )
#TODO: replace user prompt string with the following:
    #f"Given a CRISPR interference (CRISPRi) knockdown of the **{item['pert']}** gene in a single-cell **{item['cell_type']}** cell line, "
    # f"predict the regulatory effect on the **{item['gene']}** gene. "
    # "Specifically, determine if the **{item['gene']}** gene will be 'upregulated', 'downregulated', or 'not differentially expressed' "
    # "following the knockdown of **{item['pert']}**. "
        formatted_prompt = [
            {"role": "system", "content": system_prompt_string},
            {"role": "user", "content": user_prompt_string}
        ]

        
        return_dict = {
            "pert": item["pert"],
            "gene": item["gene"],  # This now holds the explanation string
            "label": item["label"],
            "cell_type": item["cell_type"],
            "prompt": formatted_prompt,
            "solution": text_solution
        }
            

        return return_dict
    