import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import warnings
import logging

# Assuming these utilities are in your project structure
from src.utils.enrichr_old import find_pathways, generate_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# TODO: Add pathway prediction functionality from another folder
# TODO: Add a mechanism for checking performance on prediction of genes expressed to synth csv

class MetabolicFluxDataset(Dataset):
    def __init__(self, csv_path, split, tokenizer=None,
                 prompt_mode: str = "default",
                 train_mode: str = "SFT",
                 test_split_cell_lines=None,
                 context: str = "none",
                 tool: str = "none",
                 exclude_sft_csv=None,
                 generate_all_non_de_samples=False,
                 generate_4x_non_de_samples=False,
                 eval_unique_genes_to_test_only=False,
                 **kwargs):
        """
        Initializes the dataset from a single input file where cell_type is a column.

        Args:
            csv_path (str): Path to the single input CSV file.
            split (str or None): The dataset split to load ('train', 'test', or None for all).
            test_split_cell_lines (str, optional): Comma-separated string of cell lines for the 'test' split.
            exclude_sft_csv (str, optional): Path to a CSV file containing samples to exclude.
            generate_all_non_de_samples (bool): If True, generates a 'not DE' (label=0) sample
                                                for every gene not explicitly listed as DE for a given perturbation.
            generate_4x_non_de_samples (bool): If True, generates 4 random non-DE samples for each DE sample.
            eval_unique_genes_to_test_only (bool): If True, ensures test set genes are not in the train set.
            **kwargs: Catches other unused arguments for flexibility.
        """
        # --- Configuration ---
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.split = split
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.context = context
        self.tool = tool

        self.test_cell_lines_list = self._parse_cell_line_list(test_split_cell_lines)
        self.use_cell_line_splitting = bool(self.test_cell_lines_list)


        self.eval_unique_genes_to_test_only = eval_unique_genes_to_test_only

        # --- Data Loading and Processing ---
        # 1. Get train genes if needed for test set uniqueness
        self.train_genes = set()
        if self.split == 'test' and self.eval_unique_genes_to_test_only:
            logging.info("Scanning for training set genes to ensure test set uniqueness...")
            self.train_genes = self._get_train_genes()
            logging.info(f"Found {len(self.train_genes)} unique genes in the training data. "
                         f"These will be excluded from the test set.")

        # 2. Load and process the single data file
        all_data_df = self._load_and_process_data()


        # 4. Final check and conversion to list of dicts
        if all_data_df.empty:
            raise RuntimeError(
                f"Dataset is empty. No data loaded for split '{self.split or 'all'}' from {self.csv_path} "
                f"with current settings."
            )

        self.data = all_data_df.to_dict('records')
        self.label_map = {0: "not changed", 1: "decreased", 2: "increased"}

        logging.info(
            f"\n--- Dataset Initialized ---\n"
            f"Split: '{self.split or 'all'}'\n"
            f"Total samples: {len(self.data)}\n"
            f"Unique cell types: {all_data_df['cell_type'].nunique()}\n"
            f"---------------------------\n"
        )

    def _load_and_process_data(self):
        """
        Loads data from the single CSV file, remaps labels, applies splits,
        and generates negative samples if configured.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Input file not found at: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise IOError(f"Error reading the CSV file at {self.csv_path}: {e}")

        # 1. Validate and Pre-process
        # CHANGE: Check for 'label' column directly
        required_cols = {'pert', 'pathway', 'cell_type', 'label', 'insulin_resistance', 'insulin_stimulation'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input CSV must contain the columns: {required_cols}")

        # CHANGE: Remap the 'label' column in place.
        # Original values: -1 (down), 0 (not DE), 1 (up)
        # Target values: 1 (down), 0 (not DE), 2 (up)
        label_mapping = {-1: 1, 0: 0, 1: 2}
        df['label'] = df['label'].map(label_mapping)
        
        # Drop rows where mapping might have failed (e.g., if there were other values in the column)
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)

        # 2. Apply Train/Test Split
        if self.use_cell_line_splitting:
            df['cell_type_lower'] = df['cell_type'].str.lower()
            if self.split == 'train':
                df_split_filtered = df[~df['cell_type_lower'].isin(self.test_cell_lines_list)]
            elif self.split == 'test':
                df_split_filtered = df[df['cell_type_lower'].isin(self.test_cell_lines_list)]
            else:
                df_split_filtered = df
            df_split_filtered = df_split_filtered.drop(columns=['cell_type_lower'])
        elif self.split and 'split' in df.columns:
            df_split_filtered = df[df['split'] == self.split]
        else:
            df_split_filtered = df

        if df_split_filtered.empty:
            warnings.warn(f"No data found for split '{self.split}' with current settings.")
            return pd.DataFrame()

        # 3. Filter Test Set for Unique Genes
        if self.split == 'test' and self.eval_unique_genes_to_test_only and self.train_genes:
            initial_count = len(df_split_filtered)
            df_split_filtered = df_split_filtered[~df_split_filtered['gene'].isin(self.train_genes)]
            if (filtered_count := initial_count - len(df_split_filtered)) > 0:
                logging.info(f"Excluded {filtered_count} test samples with genes found in the train set.")

        if df_split_filtered.empty:
            return pd.DataFrame()

        df_processed = df_split_filtered

        # 5. Final column selection
        final_cols = ['pert', 'label', 'pathway', 'cell_type', 'insulin_resistance', 'insulin_stimulation']
        return df_processed[final_cols]

    def _get_train_genes(self):
        """
        Scans the single data file to get a set of unique genes in the 'train' split.
        """
        if not self.csv_path.is_file():
            warnings.warn(f"Cannot get train genes. File not found: {self.csv_path}")
            return set()
        try:
            df_full = pd.read_csv(self.csv_path)
        except Exception as e:
            warnings.warn(f"Could not read {self.csv_path.name} to get train genes: {e}.")
            return set()

        if self.use_cell_line_splitting:
            return set(df_full[~df_full['cell_type'].str.lower().isin(self.test_cell_lines_list)]['pathway'].unique())
        elif 'split' in df_full.columns:
            return set(df_full[df_full['split'] == 'train']['pathway'].unique())
        else:
            warnings.warn("Cannot determine train split for 'eval_unique_genes_to_test_only'. "
                          "No 'split' column in CSV and 'test_split_cell_lines' not provided.")
            return set()

    def _parse_cell_line_list(self, cell_lines_str):
        """Parses a comma-separated string of cell lines into a lowercase list."""
        if not cell_lines_str or cell_lines_str.lower() in ["none", "default"]:
            return []
        return [ct.strip().lower() for ct in cell_lines_str.split(',') if ct.strip()]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_solution = self.label_map[item["label"]]
        
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
                "You are an molecular biology and metabolic systems expert analyzing changes in metabolic flux by siRNA-mediated gene silencing. "
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, gene functions. Importantly, consider these changes in the context of insulin sensitive versus resistant and whether the cells are stimulated with insulin ."
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'increased', 'decreased', or 'not changed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [increased / decreased / not changed{tool_call_option_str_2}] </answer>"
            )
        elif self.prompt_mode == "synth_data": # "o3_synth_data" in original, assuming typo in prompt
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in metabolic flux by siRNA-mediated gene silencing. "
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, gene functions. Importantly, consider these changes in the context of insulin sensitive versus resistant and whether the cells are stimulated with insulin ."
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'increased', 'decreased', or 'not changed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [increased / decreased / not changed{tool_call_option_str_2}] </answer>"
            )
        elif self.prompt_mode == "synth_data_with_ans":
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in metabolic flux by siRNA-mediated gene silencing. "
                f"The regulatory effect of knocking down the {item['pert']} gene on the {item['pathway']} gene is given to you {text_solution}."
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, gene functions. Importantly, consider these changes in the context of insulin sensitive versus resistant and whether the cells are stimulated with insulin ."
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'increased', 'decreased', or 'not changed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [increased / decreased / not changed{tool_call_option_str_2}] </answer>"

            )
        elif self.prompt_mode == "o3_test":
            system_prompt_string = (
                "You are an molecular biology and metabolic systems expert analyzing changes in metabolic flux by siRNA-mediated gene silencing. "
                "First, provide your reasoning process within <think> </think> tags. Consider the metabolic network relevant for energy metabolism, metabolic pathways, mitochondrial function, "
                "cell-type biology, ribosome biogenesis, translation machinery, oxidative stress, gene functions. Importantly, consider these changes in the context of insulin sensitive versus resistant and whether the cells are stimulated with insulin ."
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'increased', 'decreased', or 'not changed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [increased / decreased / not changed{tool_call_option_str_2}] </answer>"
            )

        user_prompt_string = (
            f"in a {item['cell_type']} cell line that is {'insulin resistant' if item['insulin_resistance'] == 1 else 'insulin sensitive'} and {'stimulated with insulin' if item['insulin_stimulation'] == 1 else 'not stimulated with insulin'}."
#TODO: replace user prompt string with the following:
#f"Given a CRISPR interference (CRISPRi) knockdown of the **{item['pert']}** gene in a single-cell **{item['cell_type']}** cell line, "
# f"predict the regulatory effect on the **{item['pathway']}** gene. "
# "Specifically, determine if the **{item['pathway']}** gene will be 'upregulated', 'downregulated', or 'not differentially expressed' "
# "following the knockdown of **{item['pert']}**. "
        )

        formatted_prompt = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]

        return_dict = {
            "pert": item["pert"],
            "gene": item["pathway"],
            "label": item["label"],
            "cell_type": item["cell_type"],
            "insulin_resistance": item["insulin_resistance"],
            "insulin_stimulation": item["insulin_stimulation"],
            "prompt": formatted_prompt,
            "solution": text_solution
        }


        return return_dict
    