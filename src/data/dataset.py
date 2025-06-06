import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np # Import numpy for efficient label calculation
import warnings # To warn about inconsistencies

from src.utils.enrichr_old import find_pathways, generate_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage, HumanMessage # To handle output type
import logging

class DiffExpressionDataset(Dataset):
    def __init__(self, csv_dir, split="train", tokenizer=None,
                train_mode:str = "SFT", prompt_mode: str = "default",
                context: str = "none", tool: str = "none",
                exclude_sft_csv: str = None,
                test_split_cell_lines: str = "none"):
        """
        Dataset for differential gene expression across multiple cell types,
        loading data from paired differential expression (-de.csv) and
        direction (-dir.csv) files.

        Args:
            csv_dir: Directory containing paired CSV files for different cell types
                     (e.g., celltypeA-de.csv, celltypeA-dir.csv).
            split: Data split to use ("train", "test", or None for all).
                   If `test_split_cell_lines` is "none", this filters based on a 'split'
                   column in the -de.csv file. Otherwise, it determines whether to load
                   the 'train' or 'test' partition defined by `test_split_cell_lines`.
            tokenizer: Tokenizer to be used, primarily for SFT mode.
            train_mode: Training mode, e.g., "SFT".
            prompt_mode: Style of prompt generation ("default", "o3_synth_data", etc.).
            context: Type of context to add ("none", "enrichr", "llm").
            tool: Tool context to potentially add ("enrichr", etc.).
            exclude_sft_csv (str, optional): Path to a CSV file containing SFT samples
                                             (pert, gene, cell_type) to exclude. Defaults to None.
            test_split_cell_lines (str, optional): Defines a cell line-based train/test splitting strategy.
                If not "none", this should be a comma-separated string of cell line names
                (e.g., "hepg2,jurkat"). These specified cell lines will constitute the 'test' set.
                All other cell lines discovered in `csv_dir` will form the 'train' set.
                This strategy overrides any 'split' column present in the input CSV files.
                The `split` parameter of this class (`__init__`) then determines whether this
                dataset instance loads the 'train' portion or the 'test' portion as defined by
                these cell line lists. If `split` is `None` (meaning load all data),
                this cell-line-based designation is effectively ignored, and all data from all
                cell lines is loaded. Defaults to "none", which means the 'split' column
                in the CSVs will be used for data partitioning if `split` is "train" or "test".
        """
        self.data = []
        self.csv_dir = Path(csv_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.context = context
        self.tool = tool

        self.test_cell_lines_list = []
        self.use_cell_line_splitting_strategy = False

        if test_split_cell_lines and test_split_cell_lines.lower() not in ["none", "default"]:
            parsed_cell_lines = [
                ct.strip().lower() for ct in test_split_cell_lines.split(',') if ct.strip()
            ]
            if parsed_cell_lines:
                self.test_cell_lines_list = parsed_cell_lines
                self.use_cell_line_splitting_strategy = True
                logging.info(
                    f"Cell line-based splitting activated. Test cell lines: {self.test_cell_lines_list}. "
                    f"Dataset instance will load for '{self.split or 'all'}' partition."
                )
            else:
                warnings.warn(
                    f"test_split_cell_lines ('{test_split_cell_lines}') was provided but resulted in an empty list. "
                    f"Falling back to CSV 'split' column-based splitting."
                )

        self.sft_exclusion_set = set()
        if exclude_sft_csv:
            exclude_sft_path = Path(exclude_sft_csv)
            if exclude_sft_path.is_file():
                try:
                    logging.info(f"Loading SFT exclusion data from: {exclude_sft_path}")
                    sft_df = pd.read_csv(exclude_sft_path)
                    required_sft_cols = {'pert', 'gene', 'cell_type'}
                    if not required_sft_cols.issubset(sft_df.columns):
                         warnings.warn(f"SFT exclusion file {exclude_sft_path} missing required columns: {required_sft_cols - set(sft_df.columns)}. Skipping exclusion.")
                    else:
                        self.sft_exclusion_set = set(
                            zip(sft_df['pert'], sft_df['gene'], sft_df['cell_type'])
                        )
                        logging.info(f"Loaded {len(self.sft_exclusion_set)} unique (pert, gene, cell_type) combinations to exclude.")
                except Exception as e:
                    warnings.warn(f"Error loading or processing SFT exclusion file {exclude_sft_path}: {e}. Skipping exclusion.")
            else:
                warnings.warn(f"SFT exclusion file not found: {exclude_sft_path}. Skipping exclusion.")

        de_files = list(self.csv_dir.glob("*-de.csv"))
        if not de_files:
            raise ValueError(f"No *-de.csv files found in {self.csv_dir}")
        logging.info(f"Found {len(de_files)} potential cell types based on -de.csv files.")

        total_samples = 0
        processed_cell_types = 0
        skipped_sft_count_total = 0 # Renamed to avoid conflict with cell_skipped_sft_count

        for de_file in de_files:
            cell_type_from_filename = de_file.stem[:-3]
            cell_type_lower = cell_type_from_filename.lower()
            dir_file = self.csv_dir / f"{cell_type_from_filename}-dir.csv"

            if not dir_file.exists():
                warnings.warn(f"Direction file {dir_file} not found for cell type {cell_type_from_filename}. Skipping this cell type.")
                continue

            try:
                df_de_full = pd.read_csv(de_file)
                df_dir = pd.read_csv(dir_file)

                required_cols_core_de = {'pert', 'gene', 'label'}
                if not required_cols_core_de.issubset(df_de_full.columns):
                    missing = required_cols_core_de - set(df_de_full.columns)
                    raise ValueError(f"Missing core columns in {de_file}: {missing}")

                required_cols_core_dir = {'pert', 'gene', 'label'}
                if not required_cols_core_dir.issubset(df_dir.columns):
                    missing = required_cols_core_dir - set(df_dir.columns)
                    raise ValueError(f"Missing core columns in {dir_file}: {missing}")

                if not self.use_cell_line_splitting_strategy and self.split and 'split' not in df_de_full.columns:
                    warnings.warn(
                        f"CSV 'split' column-based filtering was intended (split='{self.split}'), "
                        f"but 'split' column is missing in {de_file}. "
                        f"All data from this file for cell type {cell_type_from_filename} will be loaded."
                    )

                df_de_subset = pd.DataFrame()

                if self.use_cell_line_splitting_strategy:
                    is_designated_test_cell = cell_type_lower in self.test_cell_lines_list
                    load_this_cell_type = False
                    if self.split == "test":
                        if is_designated_test_cell:
                            load_this_cell_type = True
                            logging.debug(f"Cell line split: Including {cell_type_from_filename} for 'test' dataset.")
                        else:
                            logging.debug(f"Cell line split: Skipping {cell_type_from_filename} (not a designated test cell) for 'test' dataset.")
                    elif self.split == "train":
                        if not is_designated_test_cell:
                            load_this_cell_type = True
                            logging.debug(f"Cell line split: Including {cell_type_from_filename} for 'train' dataset.")
                        else:
                            logging.debug(f"Cell line split: Skipping {cell_type_from_filename} (is a designated test cell) for 'train' dataset.")
                    elif self.split is None:
                        load_this_cell_type = True
                        logging.debug(f"Cell line split (target split is None): Including ALL data from {cell_type_from_filename}.")
                    else:
                        warnings.warn(f"Unexpected self.split value '{self.split}' with cell line splitting. Skipping {cell_type_from_filename}.")
                    
                    if load_this_cell_type:
                        df_de_subset = df_de_full.copy() # Load all data from this cell line
                    else:
                        continue # Skip to next de_file
                else: # Standard CSV 'split' column-based splitting
                    if self.split and 'split' in df_de_full.columns:
                        initial_de_count = len(df_de_full)
                        df_de_subset = df_de_full[df_de_full['split'] == self.split].copy()
                        logging.info(f"Filtered {cell_type_from_filename}-de by CSV 'split' column for '{self.split}': {initial_de_count} -> {len(df_de_subset)} samples")
                    elif self.split and 'split' not in df_de_full.columns:
                        df_de_subset = df_de_full.copy()
                        logging.info(f"Loading all data from {cell_type_from_filename}-de for '{self.split}' split as 'split' column is missing.")
                    else: # self.split is None or ('split' column missing and self.split is None)
                        df_de_subset = df_de_full.copy()
                        logging.info(f"No CSV split filter applied, loading all data from {cell_type_from_filename}-de.")

                if df_de_subset.empty:
                    logging.info(f"No samples selected for cell type {cell_type_from_filename} for the current split configuration '{self.split or 'all'}'. Skipping.")
                    continue

                df_de_subset = df_de_subset.rename(columns={'label': 'is_de'})
                df_dir_renamed = df_dir.rename(columns={'label': 'direction'})

                merged_df = pd.merge(
                    df_de_subset[['pert', 'gene', 'is_de']],
                    df_dir_renamed[['pert', 'gene', 'direction']],
                    on=['pert', 'gene'],
                    how='left'
                )

                conditions = [
                    merged_df['is_de'] == 0,
                    (merged_df['is_de'] == 1) & (merged_df['direction'] == 0),
                    (merged_df['is_de'] == 1) & (merged_df['direction'] == 1)
                ]
                choices = [0, 1, 2]
                merged_df['final_label'] = np.select(conditions, choices, default=-1)

                inconsistent_rows = merged_df[
                    (merged_df['is_de'] == 1) & (merged_df['direction'].isna()) |
                    (merged_df['final_label'] == -1)
                ]
                if not inconsistent_rows.empty:
                    warnings.warn(
                        f"Found {len(inconsistent_rows)} inconsistent rows for cell type {cell_type_from_filename} "
                        f"(e.g., is_de=1 but direction is missing or invalid). Setting their label to 0 (no change). "
                        # f"Example inconsistencies:\n{inconsistent_rows.head()}" # Can be verbose
                    )
                    merged_df.loc[inconsistent_rows.index, 'final_label'] = 0
                merged_df['final_label'] = merged_df['final_label'].astype(int)

                samples_in_cell_before_exclusion = len(merged_df)
                cell_skipped_sft_count = 0
                current_cell_samples_added = 0
                for _, row in merged_df.iterrows():
                    sample_key = (row['pert'], row['gene'], cell_type_from_filename)
                    if sample_key in self.sft_exclusion_set:
                        cell_skipped_sft_count += 1
                        continue
                    self.data.append({
                        "pert": row['pert'],
                        "gene": row['gene'],
                        "label": row['final_label'],
                        "cell_type": cell_type_from_filename
                    })
                    current_cell_samples_added +=1
                
                if current_cell_samples_added > 0:
                    total_samples += current_cell_samples_added
                    processed_cell_types += 1
                skipped_sft_count_total += cell_skipped_sft_count
                logging.info(f"Processed {cell_type_from_filename}: {current_cell_samples_added} {self.split or 'total'} samples added (skipped {cell_skipped_sft_count} SFT samples).")

            except FileNotFoundError:
                warnings.warn(f"File not found during processing of {cell_type_from_filename}. Skipping.")
            except ValueError as e:
                warnings.warn(f"ValueError during processing of {cell_type_from_filename}: {e}. Skipping.")
            except Exception as e:
                 warnings.warn(f"An unexpected error occurred processing {cell_type_from_filename}: {e}. Skipping.")

        if not self.data:
             raise RuntimeError(f"Dataset is empty. No data loaded for split '{self.split or 'all'}' from {self.csv_dir} with current settings (test_split_cell_lines='{test_split_cell_lines}').")

        logging.info(f"\nTotal dataset size for split '{self.split or 'all'}': {total_samples} samples from {processed_cell_types} cell types.")
        if skipped_sft_count_total > 0:
            logging.info(f"Total SFT samples skipped across all cell types: {skipped_sft_count_total}")

        self.label_map = {0: "not differentially expressed", 1: "downregulated", 2: "upregulated"}

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
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed{tool_call_option_str_2}] </answer>"
            )
        
        elif self.prompt_mode == "o3_synth_data": # "o3_synth_data" in original, assuming typo in prompt
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                "Consider relevant pathways "
                "(e.g., cell-type biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed'."
                "When answering provide a reasoing in regulatory effect such that you use the following template: "
                " <think> </think> <answer> [upregulated / downregulated / not differentially expressed] </answer> "
                "\nExample of a CORRECT response:\n"
                "<think>\nKnocking down TF_A, a known activator of Target_Gene in this cell type, likely reduces its transcription. Relevant pathways include X and Y.\n</think>"
                "<answer>downregulated</answer>" # Corrected from 'down'
            )
        elif self.prompt_mode == "o3_synth_data_with_ans":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                f"The regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene is given to you {text_solution}. "
                "Please provide detailed resoning for your the solution by considering the following: "
                "1. Consider relevant pathways "
                "2. (e.g., cell-type biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "3 .gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', 'not differentially expressed', or 'I do not know{synth_data_tool_call}'."
                "When answering provide a reasoing in regulatory effect such that you use the following template: "
                f"<think> </think> <answer> [upregulated / downregulated / not differentially expressed / I do not know{synth_data_tool_call}] </answer> "
                "\nExample of a CORRECT response:\n"
                "<think>\nKnocking down TF_A, a known activator of Target_Gene in this cell type, likely reduces its transcription. Relevant pathways include X and Y.\n</think>"
                "<answer>downregulated</answer>"
            )
        elif self.prompt_mode == "o3_test":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                "Consider relevant pathways "
                "(e.g., cancer biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed{tool_call_option_str_2}'."
                f"Example: <answer> [upregulated / downregulated / not differentially expressed{tool_call_option_str_2}] </answer>. "
            )
        elif self.prompt_mode == "default_with_biologist_gold_solution":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed{tool_call_option_str_2}] </answer>"
            )
            gold_solution_string = (
                "\n\nHere is an example of a gold-standard response, demonstrating the expected reasoning and format:\n"
                "<think>The effect of knocking down the TNFRSF1A gene on the expression of ICAM1 in endothelial cells that had been treated with TNF is being analyzed. "
                "We expect silencing TNFRSF1A will decrease the expression of ICAM1 in response to TNF. "
                "TNFRSF1A is a receptor for TNF. The activation of TNFRSF1A by TNF induces an inflammatory response in most cell types, including endothelial cells. "
                "Such immune activation triggers the expression of cell-cell adhesion membrane proteins like ICAM1 to enable the recruitment of immune cells. "
                "If the cells were treated with TNF but the TNFRSF1A receptor is knocked down (i.e., unavailable), then the downstream signaling pathway leading to ICAM1 expression will not be properly activated. "
                "This is consistent with observations where using API antibodies inhibiting the TNF-receptor (TNFRSF1A) also inhibits inflammation. "
                "All together, the silencing of TNFRSF1A in TNF-treated endothelial cells will result in the reduced expression of ICAM1.\n"
                "</think><answer>downregulated</answer>"
            )
            system_prompt_string = system_prompt_string + gold_solution_string

        elif self.prompt_mode == "default_with_biologist_lemmas":
            print("Prompt mode: default_with_biologist_lemmas")
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed{tool_call_option_str_2}] </answer>"
            )
            lemmas_string = (
                "\n\nPlease consider the following biology specific lemmas to guide your reasoning:\n"
                "biology-specific \"lemmas\":\n"
                "1. Gene Function Lemma - What is the known molecular function of the perturbed gene product? (e.g., transcription factor, kinase, receptor, enzymes, etc.)\n"
                "2. Pathway membership Lemma – What are the molecular pathways to which the perturbed and target genes are part of?\n"
                "3. Gene perturbation phenotypic outcome Lemma – what is the known cell phenotypic outcome of either mutating, silencing, activating, or inhibiting the perturbed gene?\n"
                "4. Pathway Positioning Lemma – What is the functional role of the perturbed gene in the pathways is part of? Is the perturbed gene an activator, mediator or repressor m of any of the pathways the target gene is part of?\n"
                "5. Loci regulatory landscape Lemma – what is the regulatory landscape of the genetic loci where the target gene is located? Which transcription factor controls the target gene expression? Which cell stimuli affects the target gene expression?\n"
                "6. Direct Regulation Lemma - Is there evidence of direct gene-expression regulatory connection between the perturbed and target genes?\n"
                "7. Cell Type Context Lemma - How does the specific cell type influence the regulatory relationship between these genes?\n"
                #"8. Cell type treatment context Lemma - . Could the cellular conditions of the experiment affect the regulatory interaction betweeen the perturbed and target genes?Perturbation Mechanism Lemma - How does the specific perturbation method (CRISPR interference) affect gene expression?\n"
                "8. Analog perturbations modalities Lemma – Are there compounds or active pharmacological ingredients targeting the perturbed gene? What is the cellular response to these compounds?\n"
                "9. Temporal Dynamics Lemma - What are the expected timeframes for observing effects after perturbation?\n"
                "10. Secondary Effects Lemma - What indirect effects might occur through other intermediary genes or feedback loops?\n"
                "11. Conflicting Evidence Lemma - What contradictory evidence exists in the literature about this relationship?\n"
                "Each of these would be established with evidence and reasoning before proceeding to the final conclusion."
            )
            system_prompt_string = system_prompt_string + lemmas_string

        elif self.prompt_mode == "ans_with_biologist_lemmas":
            print("Prompt mode: ans_with_biologist_lemmas")
            first_system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                f"The regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene is given to you {text_solution}. "
            )
            lemmas_string = (
                "\n\nPlease consider the following biology specific lemmas to guide your reasoning:\n"
                "biology-specific \"lemmas\":\n"
                "1. Gene Function Lemma - What is the known molecular function of the perturbed gene product? (e.g., transcription factor, kinase, receptor, enzymes, etc.)\n"
                "2. Pathway membership Lemma – What are the molecular pathways to which the perturbed and target genes are part of?\n"
                "3. Gene perturbation phenotypic outcome Lemma – what is the known cell phenotypic outcome of either mutating, silencing, activating, or inhibiting the perturbed gene?\n"
                "4. Pathway Positioning Lemma – What is the functional role of the perturbed gene in the pathways is part of? Is the perturbed gene an activator, mediator or repressor m of any of the pathways the target gene is part of?\n"
                "5. Loci regulatory landscape Lemma – what is the regulatory landscape of the genetic loci where the target gene is located? Which transcription factor controls the target gene expression? Which cell stimuli affects the target gene expression?\n"
                "6. Direct Regulation Lemma - Is there evidence of direct gene-expression regulatory connection between the perturbed and target genes?\n"
                "7. Cell Type Context Lemma - How does the specific cell type influence the regulatory relationship between these genes?\n"
                #"8. Cell type treatment context Lemma - . Could the cellular conditions of the experiment affect the regulatory interaction betweeen the perturbed and target genes?Perturbation Mechanism Lemma - How does the specific perturbation method (CRISPR interference) affect gene expression?\n"
                "8. Analog perturbations modalities Lemma – Are there compounds or active pharmacological ingredients targeting the perturbed gene? What is the cellular response to these compounds?\n"
                "9. Temporal Dynamics Lemma - What are the expected timeframes for observing effects after perturbation?\n"
                "10. Secondary Effects Lemma - What indirect effects might occur through other intermediary genes or feedback loops?\n"
                "11. Conflicting Evidence Lemma - What contradictory evidence exists in the literature about this relationship?\n"
                "Each of these would be established with evidence and reasoning before proceeding to the final conclusion."
            )
            final_system_prompt_string = (
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', 'not differentially expressed', or 'I do not know{synth_data_tool_call}'."
                "When answering provide a reasoing in regulatory effect such that you use the following template: "
                f"<think> </think> <answer> [upregulated / downregulated / not differentially expressed / I do not know{synth_data_tool_call}] </answer> "
                "\nExample of a CORRECT response:\n"
                "<think>\nKnocking down TF_A, a known activator of Target_Gene in this cell type, likely reduces its transcription. Relevant pathways include X and Y.\n</think>"
                "<answer>downregulated</answer>"
            )
            system_prompt_string = first_system_prompt_string + lemmas_string + final_system_prompt_string


        elif self.prompt_mode == "default_with_biologist_lemmas_and_gold_solution":

            lemmas_string = (
                "\n\nPlease consider the following biology specific lemmas to guide your reasoning:\n"
                "biology-specific \"lemmas\":\n"
                "1. Gene Function Lemma - What is the known molecular function of the perturbed gene product? (e.g., transcription factor, kinase, receptor, enzymes, etc.)\n"
                "2. Pathway membership Lemma – What are the molecular pathways to which the perturbed and target genes are part of?\n"
                "3. Gene perturbation phenotypic outcome Lemma – what is the known cell phenotypic outcome of either mutating, silencing, activating, or inhibiting the perturbed gene?\n"
                "4. Pathway Positioning Lemma – What is the functional role of the perturbed gene in the pathways is part of? Is the perturbed gene an activator, mediator or repressor m of any of the pathways the target gene is part of?\n"
                "5. Loci regulatory landscape Lemma – what is the regulatory landscape of the genetic loci where the target gene is located? Which transcription factor controls the target gene expression? Which cell stimuli affects the target gene expression?\n"
                "6. Direct Regulation Lemma - Is there evidence of direct gene-expression regulatory connection between the perturbed and target genes?\n"
                "7. Cell Type Context Lemma - How does the specific cell type influence the regulatory relationship between these genes?\n"
                #"8. Cell type treatment context Lemma - . Could the cellular conditions of the experiment affect the regulatory interaction betweeen the perturbed and target genes?Perturbation Mechanism Lemma - How does the specific perturbation method (CRISPR interference) affect gene expression?\n"
                "8. Analog perturbations modalities Lemma – Are there compounds or active pharmacological ingredients targeting the perturbed gene? What is the cellular response to these compounds?\n"
                "9. Temporal Dynamics Lemma - What are the expected timeframes for observing effects after perturbation?\n"
                "10. Secondary Effects Lemma - What indirect effects might occur through other intermediary genes or feedback loops?\n"
                "11. Conflicting Evidence Lemma - What contradictory evidence exists in the literature about this relationship?\n"
                "Each of these would be established with evidence and reasoning before proceeding to the final conclusion."
            )
            gold_solution_string = (
                "\n\nHere is an example of a gold-standard response, demonstrating the expected reasoning and format:\n"
                "<think>Knocking down the TNFRSF1A gene in endothelial cells treated with TNF is expected to decrease ICAM1 expression. "
                "TNFRSF1A is a receptor for TNF, and its activation induces an inflammatory response, leading to the expression of adhesion proteins like ICAM1. "
                "If TNFRSF1A is knocked down, the signaling pathway for ICAM1 expression will be disrupted, consistent with observations where API antibodies against TNFRSF1A inhibit inflammation. "
                "Thus, silencing TNFRSF1A in TNF-treated endothelial cells will reduce ICAM1 expression.</think>"
                "<answer>downregulated</answer>"
            )
            system_prompt_string = system_prompt_string + lemmas_string + gold_solution_string
        else:
            # Default or fallback system prompt if mode is unknown\
            AssertionError (f"Unknown prompt mode: {self.prompt_mode}. Please check the prompt_mode parameter.")


        if self.prompt_mode == "direction_test_prompt":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed{tool_call_option_str_1}'."
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed{tool_call_option_str_2}] </answer>"
            )
            user_prompt_string = (
                f"It is given that the gene in question is differentially expressed either, choose one of the following options:\n"
                "1. upregulated\n"
                "2. downregulated\n"
                "Choose ONLY ONE of the options, UPREGULATED OR DOWNREGULATED, and PLACE YOUR CHOICE WITHIN <answer> </answer> TAGS.\n"
                "For this question 'not differentially expressed' is NOT an OPTION.\n"
                f" Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
            )
        elif self.prompt_mode == "o3_direction_test":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing and predicting gene regulation upon CRISPRi knockdown. "
                "Consider relevant pathways "
                "(e.g., cancer biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated' or 'downregulated'."
                f"Example: <answer> [upregulated / downregulated] </answer>. "
            )
            user_prompt_string = (
                f"It is given that the gene in question is differentially expressed either, choose one of the following options:\n"
                "1. upregulated\n"
                "2. downregulated\n"
                "Choose ONLY ONE of the options, UPREGULATED OR DOWNREGULATED, and PLACE YOUR CHOICE WITHIN <answer> </answer> TAGS.\n"
                f" Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
            )
        else: # context == "none" or other
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
            )
        if self.train_mode == "GRPO":
            prompt_string = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt_string,
                tokenize=False,
                add_generation_prompt=True # Ensures the assistant cue is added
            )
        
        else:
            formatted_prompt = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string}
            ]


        return_dict = {
            "pert": item["pert"],
            "gene": item["gene"],
            "label": item["label"],
            "cell_type": item["cell_type"],
            "prompt": formatted_prompt,
            "solution": text_solution
        }
        
        # SFT mode might require a single "text" field.
        # The original code had this commented out; keeping it as such unless specified.
        # if self.train_mode == "SFT" and self.tokenizer:
        #    # Example: full_chat = formatted_prompt + [{"role": "assistant", "content": f"<think>...</think><answer>{text_solution}</answer>"}]
        #    # The exact format for 'assistant' content depends on the expected SFT output format.
        #    # For now, assuming SFT trainer handles list of dicts. If it needs a flat string:
        #    # return_dict["text"] = self.tokenizer.apply_chat_template(full_chat, tokenize=False)
        #    # del return_dict["prompt"], return_dict["solution"] 
        #    pass

        return return_dict