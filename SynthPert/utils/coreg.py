import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List



def get_co_regulated_genes(pert: str, target_gene: str, target_label: int, cell_type: str, csv_dir: Path) -> List[str]:
    """
    Loads DE and DIR data for a cell type, finds genes co-regulated
    with the target_gene for the given pert.
    Returns an empty list if data is missing, target label is 'no change',
    or an error occurs.
    """
    if target_label == 0: # No change, no co-regulation to find based on this label
        return []

    de_file = csv_dir / f"{cell_type}-de.csv"
    dir_file = csv_dir / f"{cell_type}-dir.csv"

    if not de_file.exists() or not dir_file.exists():
        logging.warning(f"Missing de/dir file for {cell_type} while looking for co-regulated genes for {pert}/{target_gene}. Skipping enrichment.")
        return []

    try:
        df_de = pd.read_csv(de_file)
        df_dir = pd.read_csv(dir_file)

        # Filter for the specific perturbation
        df_de_pert = df_de[df_de['pert'] == pert].copy()
        df_dir_pert = df_dir[df_dir['pert'] == pert].copy()

        if df_de_pert.empty or df_dir_pert.empty:
            logging.debug(f"No data found for perturbation '{pert}' in {cell_type} files.")
            return []

        # Rename and merge (similar to Dataset logic)
        df_de_pert = df_de_pert.rename(columns={'label': 'is_de'})
        df_dir_pert = df_dir_pert.rename(columns={'label': 'direction'})

        merged_df = pd.merge(
            df_de_pert[['gene', 'is_de']],
            df_dir_pert[['gene', 'direction']],
            on='gene',
            how='inner' # Use inner merge for genes present in both for this pert
        )

        # Calculate final_label (0=no, 1=down, 2=up)
        conditions = [
            merged_df['is_de'] == 0,
            (merged_df['is_de'] == 1) & (merged_df['direction'] == 0),
            (merged_df['is_de'] == 1) & (merged_df['direction'] == 1)
        ]
        choices = [0, 1, 2]
        merged_df['final_label'] = np.select(conditions, choices, default=-1) # default=-1 for errors/inconsistencies

        # Filter for genes with the *same* regulation status as the target_label
        # Exclude the target gene itself from the co-regulated list
        co_regulated_df = merged_df[
            (merged_df['final_label'] == target_label) &
            (merged_df['gene'] != target_gene)
        ]

        co_regulated_genes = co_regulated_df['gene'].tolist()
        logging.debug(f"Found {len(co_regulated_genes)} genes co-regulated ({target_label}) with {target_gene} for pert {pert} in {cell_type}.")
        return co_regulated_genes

    except FileNotFoundError:
        logging.warning(f"FileNotFoundError processing {cell_type} for co-regulated genes for {pert}/{target_gene}. Skipping enrichment.")
        return []
    except Exception as e:
        logging.error(f"Error getting co-regulated genes for {pert}/{target_gene} in {cell_type}: {e}", exc_info=False)
        return []
