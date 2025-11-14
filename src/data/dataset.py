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
#TODO: add pathway prediction functionality to grab from other folder
#TODO: add mechanism for checking performance on prediction of genes expressed  to synth csv
class DiffExpressionDataset:
    def __init__(self, csv_dir, split, tokenizer=None,
    prompt_mode: str = "default",
    train_mode:str = "SFT",
    test_split_cell_lines=None, 
    context: str = "none", 
    tool: str = "none", 
    exclude_sft_csv=None, 
    generate_all_non_de_samples=False,
    generate_4x_non_de_samples=False,
    eval_unique_genes_to_test_only=False, 
    **kwargs):
        """
        Initializes the dataset by loading, filtering, and processing gene expression data.

        Args:
            csv_dir (str): Directory containing the '-de.csv' and '-dir.csv' files.
            split (str or None): The dataset split to load ('train', 'test', or None for all).
            test_split_cell_lines (str, optional): Comma-separated string of cell lines for the 'test' split.
            exclude_sft_csv (str, optional): Path to a CSV file containing samples to exclude.
            generate_all_non_de_samples (bool, optional): If True, generates a 'not DE' (label=0) sample
                                                          for every gene not explicitly listed as DE for a given perturbation.
                                                          Defaults to False.
            **kwargs: Catches other unused arguments for flexibility.
        """
        self.csv_dir = Path(csv_dir)
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.split = split
        self.train_mode = train_mode
        self.prompt_mode = prompt_mode
        self.context = context
        self.tool = tool

        self.test_cell_lines_list = self._parse_cell_line_list(test_split_cell_lines)
        self.use_cell_line_splitting = bool(self.test_cell_lines_list)
        # Store the new option
        self.generate_all_non_de_samples = generate_all_non_de_samples
        if self.generate_all_non_de_samples:
            logging.info("'generate_all_non_de_samples' is enabled. All non-listed genes will be included as 'not DE'.")

        self.generate_4x_non_de_samples = generate_4x_non_de_samples
        if self.generate_4x_non_de_samples:
            logging.info("'generate_4x_non_de_samples' is enabled. 4 non-DE samples will be randomly generated for each DE sample.")

        # <<< MODIFICATION: Logic to handle unique test set genes >>>
        self.eval_unique_genes_to_test_only = eval_unique_genes_to_test_only
        self.train_genes = set()
        if self.split == 'test' and self.eval_unique_genes_to_test_only:
            logging.info("Scanning for training set genes to ensure test set uniqueness...")
            self.train_genes = self._get_train_genes()
            logging.info(f"Found {len(self.train_genes)} unique genes in the training data. "
                         f"These will be excluded from the test set.")
        # <<< END MODIFICATION >>>
        
        # 1. Process all cell type files into a single, combined DataFrame
        all_data_df = self._process_all_cell_types()

        # 2. Load the SFT exclusion set
        sft_exclusions = self._load_sft_exclusion_set(exclude_sft_csv)

        # 3. Apply SFT exclusion in a single vectorized operation
        if not sft_exclusions.empty and not all_data_df.empty:
            initial_count = len(all_data_df)
            # Use a merge with an indicator to perform an "anti-join" (filter out excluded rows)
            all_data_df = all_data_df.merge(
                sft_exclusions, on=['pert', 'gene', 'cell_type'], how='left', indicator=True
            ).query('_merge == "left_only"').drop(columns=['_merge'])
            
            skipped_count = initial_count - len(all_data_df)
            if skipped_count > 0:
                logging.info(f"Total SFT samples excluded: {skipped_count}")

        # 4. Final check and conversion to the required format (list of dicts)
        if all_data_df.empty:
            raise RuntimeError(
                f"Dataset is empty. No data loaded for split '{self.split or 'all'}' from {self.csv_dir} "
                f"with current settings."
            )

        self.data = all_data_df.to_dict('records')
        self.label_map = {0: "not differentially expressed", 1: "downregulated", 2: "upregulated"}

        logging.info(
            f"\n--- Dataset Initialized ---\n"
            f"Split: '{self.split or 'all'}'\n"
            f"Total samples: {len(self.data)}\n"
            f"Unique cell types: {all_data_df['cell_type'].nunique()}\n"
            f"---------------------------\n"
        )
    # <<< MODIFICATION: Added new method to get training genes >>>
    def _get_train_genes(self):
        """
        Scans all data files to get a set of unique genes present in the 'train' split.
        This is used to filter the 'test' set for evaluation on unseen genes.
        """
        train_genes = set()
        de_files = list(self.csv_dir.glob("*-de.csv"))

        if not de_files:
            warnings.warn(f"No *-de.csv files found in {self.csv_dir} during train perturbation scan.")
            return train_genes

        for de_file in de_files:
            try:
                cell_type = de_file.stem[:-3]
                # Optimize by reading only necessary columns
                cols_to_read = ['gene', 'split'] if 'split' in pd.read_csv(de_file, nrows=0).columns else ['gene']
                df_de = pd.read_csv(de_file, usecols=cols_to_read)

                if self.use_cell_line_splitting:
                    is_test_cell_line = cell_type.lower() in self.test_cell_lines_list
                    if not is_test_cell_line: # This is a training cell line file
                        genes_to_add = df_de[df_de["label"] == 1]['gene'].unique()
                        train_genes.update(genes_to_add)
                elif 'split' in df_de.columns:
                    train_samples = df_de[df_de['split'] == 'train']
                    train_genes.update(train_samples['gene'].unique())

            except Exception as e:
                warnings.warn(f"Could not process {de_file.name} to get train genes: {e}. Skipping file.")

        return train_genes
    # <<< END MODIFICATION >>>
    def _parse_cell_line_list(self, cell_lines_str):
        """Parses a comma-separated string of cell lines into a lowercase list."""
        if not cell_lines_str or cell_lines_str.lower() in ["none", "default"]:
            return []
        return [ct.strip().lower() for ct in cell_lines_str.split(',') if ct.strip()]

    def _load_sft_exclusion_set(self, csv_path):
        """Loads SFT exclusion keys from a CSV file into a DataFrame for efficient joining."""
        if not csv_path:
            return pd.DataFrame(columns=['pert', 'gene', 'cell_type'])
        
        exclude_path = Path(csv_path)
        if not exclude_path.is_file():
            warnings.warn(f"SFT exclusion file not found: {exclude_path}. Skipping exclusion.")
            return pd.DataFrame(columns=['pert', 'gene', 'cell_type'])
        
        try:
            sft_df = pd.read_csv(exclude_path)
            required_cols = {'pert', 'gene', 'cell_type'}
            if not required_cols.issubset(sft_df.columns):
                warnings.warn(f"SFT exclusion file missing columns: {required_cols - set(sft_df.columns)}. Skipping.")
                return pd.DataFrame(columns=['pert', 'gene', 'cell_type'])
            
            return sft_df[list(required_cols)].drop_duplicates()
        except Exception as e:
            warnings.warn(f"Error loading SFT exclusion file {exclude_path}: {e}. Skipping.")
            return pd.DataFrame(columns=['pert', 'gene', 'cell_type'])

    def _process_all_cell_types(self):
        """Finds all '-de.csv' files, processes each, and concatenates them into a single DataFrame."""
        de_files = list(self.csv_dir.glob("*-de.csv"))
        if not de_files:
            raise ValueError(f"No *-de.csv files found in {self.csv_dir}")

        all_dfs = []
        for de_file in de_files:
            try:
                cell_df = self._process_single_cell_type(de_file)
                if cell_df is not None and not cell_df.empty:
                    all_dfs.append(cell_df)
            except Exception as e:
                warnings.warn(f"Failed to process {de_file.name}: {e}. Skipping file.")

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _process_single_cell_type(self, de_file):
        """
        Processes a single cell type, with an option to generate all non-DE samples.
        Returns a DataFrame for the cell type or None if it should be skipped.
        """
        cell_type = de_file.stem[:-3]
        dir_file = self.csv_dir / f"{cell_type}-dir.csv"

        if not dir_file.exists():
            warnings.warn(f"Direction file not found for {cell_type}. Skipping.")
            return None

        # This part correctly filters which *files* to process (e.g., only rpe1-de.csv)
        if self.use_cell_line_splitting:
            is_test_cell_line = cell_type.lower() in self.test_cell_lines_list
            if (self.split == 'test' and not is_test_cell_line) or \
            (self.split == 'train' and is_test_cell_line):
                return None

        # --- Data Loading and Filtering ---
        df_de = pd.read_csv(de_file)

        # Determine how to filter the data based on the splitting strategy
        if self.use_cell_line_splitting:
        # Filter for the correct split first.
            df_de_split_filtered = df_de.copy()
        
        else:
        # Filter by the 'split' column if a split is provided,
            if self.split and 'split' in df_de.columns:
                df_de_split_filtered = df_de[df_de['split'] == self.split]
            else:
                df_de_split_filtered = df_de.copy()

        # <<< MODIFICATION: Filter test set for unique perturbations >>>
        if self.split == 'test' and self.eval_unique_genes_to_test_only:
            if not df_de_split_filtered.empty and self.train_perts:
                initial_count = len(df_de_split_filtered)
                df_de_split_filtered = df_de_split_filtered[~df_de_split_filtered['pert'].isin(self.train_perts)]
                filtered_count = initial_count - len(df_de_split_filtered)
                if filtered_count > 0:
                    logging.info(f"[{cell_type}] Excluded {filtered_count} samples belonging to perturbations also found in the train set.")
        # <<< END MODIFICATION >>>

        if df_de_split_filtered.empty:
            return None

        # --- Generate Negative Samples if Option is Enabled ---
        if self.generate_all_non_de_samples:
            all_genes = df_de['gene'].unique()
            perts_in_split = df_de_split_filtered['pert'].unique()
            
            all_possible_pairs = pd.MultiIndex.from_product(
                [perts_in_split, all_genes], names=['pert', 'gene']
            ).to_frame(index=False)

            df_de_processed = pd.merge(
                all_possible_pairs,
                df_de_split_filtered[['pert', 'gene', 'label']],
                on=['pert', 'gene'],
                how='left'
            )
            
            # <<< FIX for FutureWarning: Use direct assignment >>>
            df_de_processed['label'] = df_de_processed['label'].fillna(0)

        elif self.generate_4x_non_de_samples:
            # Get all unique genes available in the full dataset for this cell type.
            all_genes = set(df_de['gene'].unique())
            # Isolate only the differentially expressed (DE) samples from the current split.
            de_samples_in_split = df_de_split_filtered[df_de_split_filtered['label'] > 0]

            if de_samples_in_split.empty:
                # If there are no DE genes, we don't need to add any negative samples.
                df_de_processed = df_de_split_filtered.copy()
            else:
                # Define a simple function to perform the sampling for each perturbation group.
                def sample_negatives(group):
                    # Find which genes are DE for this specific perturbation.
                    de_genes_for_pert = set(group['gene'])
                    # Determine the pool of genes that are NOT DE for this perturbation.
                    non_de_pool = list(all_genes - de_genes_for_pert)
                    
                    # Calculate how many negative samples to draw (4x the number of DE genes).
                    n_to_sample = len(de_genes_for_pert) * 4
                    
                    # Don't sample more genes than are available in the pool.
                    actual_n_to_sample = min(n_to_sample, len(non_de_pool))
                    
                    # Randomly choose genes from the non-DE pool.
                    sampled_genes = np.random.choice(non_de_pool, size=actual_n_to_sample, replace=False)
                    
                    # Return a new DataFrame for these negative samples.
                    return pd.DataFrame({
                        'gene': sampled_genes,
                        'pert': group.name, # .name contains the perturbation ID when using groupby
                        'label': 0
                    })

                # Apply the sampling function to each perturbation group and combine the results.
                new_neg_samples = de_samples_in_split.groupby('pert').apply(sample_negatives).reset_index(drop=True)

                # Combine the original data with the newly generated negative samples.
                combined_df = pd.concat([df_de_split_filtered, new_neg_samples], ignore_index=True)
                
                # In case a sampled gene was already in the original data (as label 0),
                # drop the duplicate, keeping the original.
                df_de_processed = combined_df.drop_duplicates(subset=['pert', 'gene'], keep='first')


        else:
            df_de_processed = df_de_split_filtered
        
        # --- Merging and Final Label Creation ---
        df_dir = pd.read_csv(dir_file)
        merged = pd.merge(
            df_de_processed.rename(columns={'label': 'is_de'}),
            df_dir.rename(columns={'label': 'direction'})[['pert', 'gene', 'direction']],
            on=['pert', 'gene'], how='left'
        )

        conditions = [
            merged['is_de'] == 0,
            (merged['is_de'] == 1) & (merged['direction'] == 0),
            (merged['is_de'] == 1) & (merged['direction'] == 1)
        ]
        choices = [0, 1, 2]
        
        merged['label'] = np.select(conditions, choices, default=0).astype(int)
        merged['cell_type'] = cell_type

        return merged[['pert', 'gene', 'label', 'cell_type']]

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
        elif self.prompt_mode == "o3_syth_data": # "o3_synth_data" in original, assuming typo in prompt
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


        if self.context=="enrichr":
            sample_genes = [item["gene"], item["pert"]]
            pathways_context = find_pathways(sample_genes)
            prompt_context_data = generate_prompt(pathways_context, sample_genes)
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
                f"I am providing gene enrichment analysis data from Enrichr related to the {item['gene']} gene and the "
                f"knocked down {item['pert']} gene, which may inform your prediction:\n{prompt_context_data}"
            )
        elif self.context=="llm":
            # It's generally better practice to initialize the LLM client once in __init__
            # or pass it as an argument, rather than creating it for each __getitem__ call.
            # However, following the provided structure.
            def get_llm_context(pert, gene, cell_type):
                # IMPORTANT: Hardcoding API keys is a security risk. Use environment variables or a config file.
                # This key is likely a placeholder or example.
                api_key = os.getenv("OPENAI_API_KEY", "xxxxx") # Example: use env var or default
                if api_key == "xxxxx" and not os.getenv("OPENAI_API_KEY"):
                    logging.warning("Using a hardcoded placeholder API key for LLM context.")

                llm_for_context = ChatOpenAI(
                    model="openai_o3_mini", # Ensure this model name is correct for the endpoint
                    api_key=api_key,
                    base_url="xxxx", # Ensure this endpoint is correct
                    max_retries=3,
                    request_timeout=120,
                )
                system_prompt_text = (
                        "You are a knowledgeable bioinformatics expert specializing in gene regulation and pathway analysis. "
                        "Your task is to retrieve and synthesize information about the functional relationship between genes, "
                        "particularly in the context of perturbations and specific cell types. Focus on known pathways, biological processes, "
                        "and molecular functions relevant to the user's query. Provide a concise summary suitable as background context "
                        "for analyzing regulatory effects."
                    )
                user_query_text = (
                    f"Summarize the key functional annotations, pathways (e.g., KEGG, GO terms, Reactome), and biological processes "
                    f"associated with the interplay between the gene '{pert}' (specifically when perturbed, like knockdown/knockout) "
                    f"and the target gene '{gene}' within the context of '{cell_type}' cells. "
                    f"Highlight information relevant to understanding potential regulatory effects of perturbing '{pert}' on '{gene}'."
                )
                messages = [
                    SystemMessage(content=system_prompt_text),
                    HumanMessage(content=user_query_text),
                ]
                try:
                    response = llm_for_context.invoke(messages)
                    if isinstance(response, AIMessage):
                        return "LLM (o3_mini)", response.content
                    else:
                        logging.error(f"Unexpected response type from LLM: {type(response)}")
                        return "LLM", "Error: Unexpected response type."
                except Exception as e:
                    logging.error(f"Error invoking LLM for context: {e}")
                    return "LLM", f"Error retrieving context: {e}"

            model_name, llm_context_data = get_llm_context(item["pert"], item["gene"], item["cell_type"])
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
                f"I am providing gene enrichment analysis from {model_name} related to the {item['gene']} gene and the "
                f"knocked down {item['pert']} gene, which may inform your prediction:\n{llm_context_data}"
            )
        elif self.prompt_mode == "direction_test_prompt":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated' or 'downregulated'."
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated] </answer>"
            )
            user_prompt_string = (
                f"It is given that the gene in question is differentially expressed either, choose one of the following options:\n"
                "1. upregulated\n"
                "2. downregulated\n"
                "CHOSE ONE OF THE ABOVE OPTIONS AND PLACE YOUR CHOICE WITHIN <answer> </answer> TAGS.\n"
                f" Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
            )
        elif self.prompt_mode == "warning_different_distributions":
            system_prompt_string = (
                "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed'. "
                "Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed] </answer>"
            )
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
                "It is important to note that the dataset from which this observation is drawn contains a higher-than-expected proportion of genes "
                "classified as 'not differentially expressed'. Please account for this distributional characteristic in your prediction, "
                "but do not let it override strong biological evidence for differential expression."
            )
        elif self.prompt_mode == "gene_enrichment":
            system_prompt_string = (
                "You are a molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
                "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
                "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
                "gene interactions, and cell-specific context. "
                f"Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed'. "
                f"Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed] </answer>"
            )
            pathway = item['gene'].split("-")[-1]
            user_prompt_string = (
                f"Analyze the gene regulation of knocking down the {item['pert']} gene on the {pathway} pathway "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference. "
                "Please provide a detailed analysis of the pathway interactions and regulatory effects. "
            )
        else: # context == "none" or other
            warnings.warn(f"Unknown prompt_mode: {self.prompt_mode}. Using a default system prompt.")
            user_prompt_string = (
                f"Analyze the regulatory effect of knocking down the {item['pert']} gene on the {item['gene']} gene "
                f"in a single-cell {item['cell_type']} cell line using CRISPR interference."
    #TODO: replace user prompt string with the following:
    #f"Given a CRISPR interference (CRISPRi) knockdown of the **{item['pert']}** gene in a single-cell **{item['cell_type']}** cell line, "
    # f"predict the regulatory effect on the **{item['gene']}** gene. "
    # "Specifically, determine if the **{item['gene']}** gene will be 'upregulated', 'downregulated', or 'not differentially expressed' "
    # "following the knockdown of **{item['pert']}**. "
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
        if self.prompt_mode == "gene_enrichment":
            pathway = item['gene'].split("-")[-1]
            return_dict = {
                "pert": item["pert"],
                "gene": pathway,
                "label": item["label"],
                "cell_type": item["cell_type"],
                "prompt": formatted_prompt,
                "solution": text_solution
            }
        else:
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
    
