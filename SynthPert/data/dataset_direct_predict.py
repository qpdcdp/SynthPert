import os
import pandas as pd
import torch # Not strictly used by this class directly, but common for Dataset parent
from torch.utils.data import Dataset
from pathlib import Path
import warnings
import logging

# Configure basic logging if not already set up by the environment
# (e.g., when running this script directly for testing)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneRegulationListDataset(Dataset):
    def __init__(self, csv_dir, split="train", tokenizer=None,
                 train_mode: str = "SFT",
                 prompt_mode: str = "default",
                 test_split_cell_lines: str = "none"):
        """
        Dataset for predicting lists of upregulated and downregulated genes
        given a perturbation and cell type. Data is sourced from '*-dir.csv' files.

        Each sample in this dataset corresponds to a (perturbation, cell_type) pair,
        and the target is a list of upregulated genes and a list of downregulated genes.

        Args:
            csv_dir (str or Path): Directory containing '*-dir.csv' files
                                   (e.g., celltypeA-dir.csv).
            split (str, optional): Data split to use ("train", "test", or None for all).
                                   Defaults to "train". If `test_split_cell_lines` is "none",
                                   this filters based on a 'split' column in the -dir.csv files.
                                   Otherwise, it determines whether to load the 'train' or 'test'
                                   partition defined by `test_split_cell_lines`.
            tokenizer (optional): Tokenizer to be used, primarily for formatting prompts
                                  for SFT mode.
            train_mode (str, optional): Training mode, e.g., "SFT", "GRPO".
                                        Affects prompt structure returned by __getitem__.
                                        Defaults to "SFT".
            prompt_mode (str, optional): Style of prompt generation.
                                         Defaults to "default_list".
            test_split_cell_lines (str, optional): Defines a cell line-based train/test
                                                   splitting strategy. If not "none", this
                                                   should be a comma-separated string of
                                                   cell line names (e.g., "hepg2,jurkat").
                                                   These specified cell lines will constitute
                                                   the 'test' set. All other cell lines
                                                   discovered in `csv_dir` will form the
                                                   'train' set. This strategy overrides any
                                                   'split' column present in the input CSV
                                                   files for the purpose of selecting which
                                                   cell lines belong to train/test.
                                                   Defaults to "none".
        """
        self.data = []
        self.csv_dir = Path(csv_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode

        self.parsed_test_cell_lines_lc = [] # Lowercase designated test cell lines
        self.use_cell_line_splitting_strategy = False

        if test_split_cell_lines and test_split_cell_lines.lower() != "none":
            raw_test_lines = [ct.strip().lower() for ct in test_split_cell_lines.split(',') if ct.strip()]
            if raw_test_lines:
                self.parsed_test_cell_lines_lc = raw_test_lines
                self.use_cell_line_splitting_strategy = True
                logging.info(
                    f"Cell line-based splitting activated. Designated test cell lines (lowercase): {self.parsed_test_cell_lines_lc}. "
                    f"Dataset instance will load for '{self.split or 'all data'}' partition."
                )
            else:
                warnings.warn(
                    f"test_split_cell_lines ('{test_split_cell_lines}') was provided but resulted in an empty list. "
                    f"Falling back to CSV 'split' column-based splitting for row filtering."
                )

        dir_files = sorted(list(self.csv_dir.glob("*-dir.csv"))) # Sort for deterministic order
        if not dir_files:
            raise ValueError(f"No '*-dir.csv' files found in {self.csv_dir}")
        logging.info(f"Found {len(dir_files)} '*-dir.csv' files in {self.csv_dir}.")

        all_discovered_cell_types_lc = sorted(list(set(
            f.stem[:-4].lower() for f in dir_files # Assumes format 'celltype-dir.csv', e.g. 'hepg2-dir' -> 'hepg2'
        )))
        
        cell_types_to_process_lc = [] # List of lowercase cell type names to actually load files for

        if self.use_cell_line_splitting_strategy:
            # Determine effective train/test sets based on discovered cell types
            strategy_effective_test_set_lc = {ct_lc for ct_lc in all_discovered_cell_types_lc if ct_lc in self.parsed_test_cell_lines_lc}
            strategy_effective_train_set_lc = {ct_lc for ct_lc in all_discovered_cell_types_lc if ct_lc not in strategy_effective_test_set_lc}
            
            logging.info(f"Cell line strategy: Effective train cell types from discovered files: {list(strategy_effective_train_set_lc)}")
            logging.info(f"Cell line strategy: Effective test cell types from discovered files: {list(strategy_effective_test_set_lc)}")

            if self.split == "train":
                cell_types_to_process_lc = list(strategy_effective_train_set_lc)
            elif self.split == "test":
                cell_types_to_process_lc = list(strategy_effective_test_set_lc)
            elif self.split is None: 
                cell_types_to_process_lc = all_discovered_cell_types_lc
            else:
                warnings.warn(f"Unexpected self.split value '{self.split}' with cell line splitting strategy. No cell types will be loaded.")
        else:
            # No cell line strategy: all discovered cell types are candidates.
            # Filtering by 'split' column will happen row-wise within each file.
            cell_types_to_process_lc = all_discovered_cell_types_lc

        logging.info(f"Will attempt to load data for '{self.split or 'all data'}' split from {len(cell_types_to_process_lc)} cell types: {cell_types_to_process_lc}")

        total_pert_cell_samples_added = 0
        processed_cell_type_file_count = 0

        for dir_file_path in dir_files:
            # Original filename casing is preserved for 'cell_type' field in self.data
            cell_type_original_casing = dir_file_path.stem[:-4] 
            cell_type_lc = cell_type_original_casing.lower()

            if cell_type_lc not in cell_types_to_process_lc:
                logging.debug(f"Skipping file {dir_file_path} as cell type '{cell_type_lc}' is not in the target list for the current split configuration.")
                continue
            
            logging.info(f"Processing file: {dir_file_path} for cell type '{cell_type_original_casing}'")
            processed_cell_type_file_count += 1

            try:
                df_full_cell = pd.read_csv(dir_file_path)
                required_cols = {'pert', 'gene', 'label'} # 'split' is optional
                if not required_cols.issubset(df_full_cell.columns):
                    missing = required_cols - set(df_full_cell.columns)
                    warnings.warn(f"Missing core columns in {dir_file_path}: {missing}. Skipping this file.")
                    continue

                df_for_grouping = df_full_cell # Start with all data from the file

                if not self.use_cell_line_splitting_strategy:
                    # Apply internal 'split' column filtering only if not using cell line strategy
                    if self.split and 'split' not in df_full_cell.columns:
                        warnings.warn(
                            f"CSV 'split' column filtering intended for '{self.split}' split "
                            f"(as test_split_cell_lines is 'none' or inactive), but 'split' column missing in {dir_file_path}. "
                            f"All data from this file will be used for grouping for cell type {cell_type_original_casing}."
                        )
                    elif self.split and 'split' in df_full_cell.columns:
                        initial_row_count = len(df_full_cell)
                        df_for_grouping = df_full_cell[df_full_cell['split'] == self.split].copy()
                        logging.info(
                            f"Filtered {dir_file_path} by CSV 'split' column for '{self.split}': {initial_row_count} -> {len(df_for_grouping)} rows."
                        )
                # If use_cell_line_splitting_strategy is True, or if self.split is None,
                # df_for_grouping remains df_full_cell (all rows from the file are used),
                # as cell type file selection has already determined its inclusion in train/test/all.

                if df_for_grouping.empty:
                    logging.info(f"No data rows selected from {dir_file_path} after split filtering (if any). No samples generated from this file.")
                    continue

                grouped_by_pert = df_for_grouping.groupby('pert')
                
                num_pert_groups_in_file = 0
                for pert, group_df in grouped_by_pert:
                    # Ensure gene names are strings, handle potential NaN/float if data is malformed
                    up_genes = sorted(list(set(
                        str(g) for g in group_df[group_df['label'] == 1]['gene'].tolist() if pd.notna(g)
                    )))
                    down_genes = sorted(list(set(
                        str(g) for g in group_df[group_df['label'] == 0]['gene'].tolist() if pd.notna(g)
                    )))

                    self.data.append({
                        "pert": pert,
                        "cell_type": cell_type_original_casing,
                        "upregulated_genes": up_genes,
                        "downregulated_genes": down_genes
                    })
                    num_pert_groups_in_file += 1
                
                if num_pert_groups_in_file > 0:
                    logging.info(f"Generated {num_pert_groups_in_file} (pert, cell_type) samples from {dir_file_path}.")
                    total_pert_cell_samples_added += num_pert_groups_in_file
                else:
                    logging.info(f"Although {dir_file_path} was processed, no (pert, cell_type) samples were generated (e.g., empty after grouping).")

            except FileNotFoundError:
                warnings.warn(f"File disappeared during processing: {dir_file_path}. Skipping.") # Should be rare
            except pd.errors.EmptyDataError:
                warnings.warn(f"CSV file {dir_file_path} is empty. Skipping.")
            except Exception as e:
                 warnings.warn(f"An unexpected error occurred processing {dir_file_path}: {e}. Skipping.")

        if not self.data:
            reason_message = "Specific reason undetermined."
            if not dir_files: # Should have been caught earlier
                reason_message = "no '*-dir.csv' files were found in the directory."
            elif not all_discovered_cell_types_lc:
                 reason_message = "no cell types could be discovered from filenames."
            elif not cell_types_to_process_lc :
                 reason_message = (f"no cell types were selected to be processed for the '{self.split or 'all data'}' split "
                                   f"given test_split_cell_lines='{test_split_cell_lines}' and discovered files.")
            elif processed_cell_type_file_count == 0 and len(cell_types_to_process_lc) > 0 :
                 reason_message = "files for selected cell types might be missing or unreadable." # Should be caught by file loop
            elif total_pert_cell_samples_added == 0 and processed_cell_type_file_count > 0:
                 reason_message = "processed files yielded no (pert, cell_type) groups with gene data after filtering and grouping."
            
            raise RuntimeError(
                f"Dataset is empty. No data loaded for split '{self.split or 'all data'}' from {self.csv_dir}. "
                f"Reason: {reason_message}"
            )

        logging.info(
            f"\nDataset initialization complete for split '{self.split or 'all data'}'.\n"
            f"Total (pert, cell_type) samples loaded: {len(self.data)}\n"
            f"Processed {processed_cell_type_file_count} '*-dir.csv' file(s) corresponding to selected cell types."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format the solution string content (Python list __str__ method is convenient here)
        solution_content_up_list = f"Upregulated: {item['upregulated_genes']}"
        solution_content_down_list = f"Downregulated: {item['downregulated_genes']}"
        # Combine and wrap in <answer> tags for the final solution text
        text_solution_payload = f"{solution_content_up_list}\n{solution_content_down_list}"
        final_solution_text = f"<answer>{text_solution_payload}</answer>"

        system_prompt_string = ""
        if self.prompt_mode == "default":
            system_prompt_string = (
                "You are an expert bioinformatician. Given a gene perturbation and a cell type, "
                "your task is to list all genes that are upregulated and all genes that are downregulated "
                "as a result of this perturbation.\n"
                "The list of upregulated genes should be prefixed with 'Upregulated: '.\n"
                "The list of downregulated genes should be prefixed with 'Downregulated: '.\n"
                "Each list of genes should be seperated by a semi-colon and a space.\n"
                "Gene names within each list should be enclosed in square brackets and separated by a comma and a space, e.g., ['GENE_A', 'GENE_B'].\n"
                "If no genes fall into a category, use an empty list: [].\n"
                "Wrap your entire response, starting with 'Upregulated:', within <answer> </answer> tags.\n\n"
                "Example of a CORRECT response with genes in both categories:\n"
                "<think>[Your reasoning here]</think><answer>Upregulated: ['GENE_A', 'GENE_B']; Downregulated: ['GENE_C']</answer>\n\n"
                "Example of a CORRECT response with only upregulated genes:\n"
                "<think>[Your reasoning here]</think><answer>Upregulated: ['GENE_X', 'GENE_Y']; Downregulated: []</answer>\n\n"
                "Example of a CORRECT response with no genes in either category:\n"
                "<think>[Your reasoning here]</think><answer>Upregulated: []; Downregulated: []</answer>"
            )
        elif self.prompt_mode == "synth_data_with_ans":
            system_prompt_string = (
                "You are an expert bioinformatician analyzing and predicting gene regulation upon CRISPRi knockdown. "
                f"The regulatory effect of knocking down the {item['pert']} gene  is given to you: {text_solution_payload}. "
                "Please provide detailed resoning for your the solution by considering the following: "
                "1. Consider relevant pathways "
                "2. (e.g., cell-type biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "3 .gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your answer within <answer> </answer> tags."
                "When answering provide a reasoing in regulatory effect such that you use the following template: "
                f"<think> </think> <answer></answer> "
                "\nExample of a CORRECT response:\n"
                "<think>\nKnocking down TF_A, a known activator of Target_Gene in this cell type, likely reduces its transcription. Relevant pathways include X and Y.\n</think><answer>Upregulated: ['GENE_A', 'GENE_B']; Downregulated: ['GENE_C']</answer></answer>"
            )
        else:
            warnings.warn(f"Unknown prompt_mode: {self.prompt_mode}. Using a generic system prompt.")
            system_prompt_string = ("You are a helpful assistant. Please list upregulated and downregulated genes based on the user query. "
                                    "Format upregulated genes as 'Upregulated: [GENE_LIST]' and downregulated genes as 'Downregulated: [GENE_LIST]', each on a new line, "
                                    "and wrap the entire response in <answer></answer> tags.")

        user_prompt_string = (
            f"For the perturbation of gene {item['pert']} in {item['cell_type']} cells, "
            "identify the upregulated and downregulated genes according to the specified format."
        )

        formatted_prompt_input: any # Can be list of dicts or string

        if self.train_mode == "SFT" and self.tokenizer:
            # For SFT, the prompt is typically a string that the model completes.
            # apply_chat_template with add_generation_prompt=True creates this input string.
            prompt_messages = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]
            formatted_prompt_input = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True # Crucial for signaling assistant's turn
            )
        else: # For "GRPO" or if tokenizer is not available for SFT.
              # Default to list of message dicts.
            if self.train_mode == "SFT" and not self.tokenizer:
                warnings.warn("SFT mode selected but no tokenizer provided. Prompt will be a list of dicts.")
            
            formatted_prompt_input = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]

        return_dict = {
            "pert": item["pert"],
            "cell_type": item["cell_type"],
            "prompt": formatted_prompt_input, # Input to the model
            "solution_text": final_solution_text, # Target output from the model
            "raw_solution_lists": { # For easier non-text based evaluation or use
                "upregulated": item['upregulated_genes'],
                "downregulated": item['downregulated_genes']
            }
        }
        
        return return_dict