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

class PerturbationPredictionDataset(Dataset):
    def __init__(self, csv_dir, split="train", tokenizer=None,
                 train_mode: str = "none",
                 prompt_mode: str = "default",
                 list_format: str = "default", # This can still be used to format the *input* gene lists
                 test_split_cell_lines: str = "none"):
        """
        Dataset for predicting the source perturbation (gene) given lists of
        upregulated and downregulated genes and a cell type. Data is sourced
        from '*-dir.csv' files.

        Each sample in this dataset corresponds to a (list of up genes, list of down genes)
        and the target is the perturbation that caused these changes.
        """
        # --- This __init__ method requires no changes ---
        # The logic for finding, loading, and grouping data by perturbation
        # is still exactly what we need. Each item in self.data will
        # still be a dictionary containing the pert, cell_type, and gene lists.
        self.data = []
        self.csv_dir = Path(csv_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.list_format = list_format

        self.parsed_test_cell_lines_lc = []
        self.use_cell_line_splitting_strategy = False

        if test_split_cell_lines and test_split_cell_lines.lower() not in ["none", "default"]:
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

        dir_files = sorted(list(self.csv_dir.glob("*-dir.csv")))
        if not dir_files:
            raise ValueError(f"No '*-dir.csv' files found in {self.csv_dir}")
        logging.info(f"Found {len(dir_files)} '*-dir.csv' files in {self.csv_dir}.")

        all_discovered_cell_types_lc = sorted(list(set(
            f.stem[:-4].lower() for f in dir_files
        )))
        
        cell_types_to_process_lc = []

        if self.use_cell_line_splitting_strategy:
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
            cell_types_to_process_lc = all_discovered_cell_types_lc

        logging.info(f"Will attempt to load data for '{self.split or 'all data'}' split from {len(cell_types_to_process_lc)} cell types: {cell_types_to_process_lc}")

        total_pert_cell_samples_added = 0
        processed_cell_type_file_count = 0

        for dir_file_path in dir_files:
            cell_type_original_casing = dir_file_path.stem[:-4] 
            cell_type_lc = cell_type_original_casing.lower()

            if cell_type_lc not in cell_types_to_process_lc:
                logging.debug(f"Skipping file {dir_file_path} as cell type '{cell_type_lc}' is not in the target list for the current split configuration.")
                continue
            
            logging.info(f"Processing file: {dir_file_path} for cell type '{cell_type_original_casing}'")
            processed_cell_type_file_count += 1

            try:
                df_full_cell = pd.read_csv(dir_file_path)
                required_cols = {'pert', 'gene', 'label'}
                if not required_cols.issubset(df_full_cell.columns):
                    missing = required_cols - set(df_full_cell.columns)
                    warnings.warn(f"Missing core columns in {dir_file_path}: {missing}. Skipping this file.")
                    continue

                df_for_grouping = df_full_cell

                if not self.use_cell_line_splitting_strategy:
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
                
                if df_for_grouping.empty:
                    logging.info(f"No data rows selected from {dir_file_path} after split filtering (if any). No samples generated from this file.")
                    continue

                grouped_by_pert = df_for_grouping.groupby('pert')
                
                num_pert_groups_in_file = 0
                for pert, group_df in grouped_by_pert:
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
                warnings.warn(f"File disappeared during processing: {dir_file_path}. Skipping.")
            except pd.errors.EmptyDataError:
                warnings.warn(f"CSV file {dir_file_path} is empty. Skipping.")
            except Exception as e:
                 warnings.warn(f"An unexpected error occurred processing {dir_file_path}: {e}. Skipping.")

        if not self.data:
            reason_message = "Specific reason undetermined."
            if not dir_files:
                reason_message = "no '*-dir.csv' files were found in the directory."
            elif not all_discovered_cell_types_lc:
                 reason_message = "no cell types could be discovered from filenames."
            elif not cell_types_to_process_lc :
                 reason_message = (f"no cell types were selected to be processed for the '{self.split or 'all data'}' split "
                                   f"given test_split_cell_lines='{test_split_cell_lines}' and discovered files.")
            elif processed_cell_type_file_count == 0 and len(cell_types_to_process_lc) > 0 :
                 reason_message = "files for selected cell types might be missing or unreadable."
            elif total_pert_cell_samples_added == 0 and processed_cell_type_file_count > 0:
                 reason_message = "processed files yielded no (pert, cell_type) groups with gene data after filtering and grouping."
            
            raise RuntimeError(
                f"Dataset is empty. No data loaded for split '{self.split or 'all data'}' from {self.csv_dir}. "
                f"Reason: {reason_message}"
            )

        logging.info(
            f"\nDataset initialization complete for split '{self.split or 'all data'}'.\n"
            f"Total samples loaded: {len(self.data)}\n"
            f"Processed {processed_cell_type_file_count} '*-dir.csv' file(s) corresponding to selected cell types."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # <<< --- START OF MAJOR CHANGES --- >>>

        # <<< CHANGE 1: The "solution" is now the perturbation gene.
        # This is the target for the model to predict.
        final_solution_text = f"{item['pert']}"

        # <<< CHANGE 2: The gene lists are now part of the INPUT prompt.
        # We format them here to be included in the user_prompt_string.
        # We can reuse the self.list_format logic for this.
        if self.list_format == "default":
            # Format using Python's default list __str__ representation
            up_genes_str = str(item['upregulated_genes'])
            down_genes_str = str(item['downregulated_genes'])
            gene_list_input_text = f"Upregulated genes: {up_genes_str}\nDownregulated genes: {down_genes_str}"

        elif self.list_format == "bullet_list":
            # Format as bulleted lists for clarity in the prompt
            up_genes_list = "\n".join([f"- {gene}" for gene in item['upregulated_genes']]) if item['upregulated_genes'] else "NONE"
            down_genes_list = "\n".join([f"- {gene}" for gene in item['downregulated_genes']]) if item['downregulated_genes'] else "NONE"
            gene_list_input_text = f"UPREGULATED_GENES:\n{up_genes_list}\n\nDOWNREGULATED_GENES:\n{down_genes_list}"
        
        else: # Fallback for any other list_format
            up_genes_str = str(item['upregulated_genes'])
            down_genes_str = str(item['downregulated_genes'])
            gene_list_input_text = f"Upregulated: {up_genes_str}\nDownregulated: {down_genes_str}"

        if self.prompt_mode == "synth_data_with_ans":
            system_prompt_string = (
                "You are a molecular and cellular biology expert. "
                f"You are given that knocking down the gene '{item['pert']}' in {item['cell_type']} cells "
                "results in a specific set of gene expression changes.\n\n"
                "Your task is to provide detailed reasoning for why this perturbation causes these effects. "
                "Consider relevant pathways, gene interactions, and cell-specific context. "
                "After your reasoning, state the name of the perturbed gene in the answer tag.\n\n"
                "Use the following template:\n"
                "<think>\n[Your detailed reasoning about why knocking down this gene leads to the observed changes goes here.]\n</think>"
                "<answer>[The name of the gene that was knocked down]</answer>"
            )

            user_prompt_string = (
                f"In {item['cell_type']} cells, a CRISPRi knockdown led to the following changes in gene expression:\n\n"
                f"{gene_list_input_text}\n\n"
                "Provide your detailed reasoning and then identify the gene that was knocked down."
            )

        else:
            system_prompt_string = (
                "You are a molecular and cellular biology expert. "
                "Your task is to identify the single gene that was knocked down via CRISPRi "
                "based on the resulting lists of upregulated and downregulated genes in a specific cell type.\n\n"
                "Analyze the provided gene lists and the cell context to infer the most likely "
                "causative gene perturbation. Your answer must be only the name of the gene.\n\n"
                "Wrap your entire response within <answer> </answer> tags.\n\n"
                "Example of a CORRECT response:\n"
                "<think>[Your reasoning here]</think><answer>GENE_X</answer>"
            )

            user_prompt_string = (
                f"In {item['cell_type']} cells, a single gene was knocked down using CRISPRi, "
                "leading to the following changes in gene expression:\n\n"
                f"{gene_list_input_text}\n\n"
                "Based on this data, what is the most likely gene that was knocked down?"
            )
        # The logic for formatting the prompt for SFT vs. other modes remains the same.
        formatted_prompt_input: any

        if self.train_mode == "SFT" and self.tokenizer:
            prompt_messages = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]
            formatted_prompt_input = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            if self.train_mode == "SFT" and not self.tokenizer:
                warnings.warn("SFT mode selected but no tokenizer provided. Prompt will be a list of dicts.")
            
            formatted_prompt_input = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]

        # <<< CHANGE 4: The returned dictionary is updated to reflect the new structure.
        # "pert" is now part of the solution, not a separate key.
        return_dict = {
            "cell_type": item["cell_type"],
            "prompt": formatted_prompt_input,      # Input to the model (contains gene lists)
            "solution": final_solution_text,      # Target output from the model (the pert)
            "raw_solution_pert": item['pert'],    # The raw target value
            "raw_input_lists": {                  # The raw input values for reference
                "upregulated": item['upregulated_genes'],
                "downregulated": item['downregulated_genes']
            }
        }
        
        return return_dict

        # <<< --- END OF MAJOR CHANGES --- >>>