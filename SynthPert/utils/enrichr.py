import pandas as pd
import enrichrpy.enrichr as een
import logging
import time # Keep for potential delays if needed

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Gene Set Libraries (Keep your selection or update as needed) ---
# Check available libraries: print(een.get_libraries())
# Ensure the names match exactly what enrichrpy expects.
# Let's verify some common ones exist:
try:
    available_libs = een.get_libraries()
    logging.info(f"Total available libraries: {len(available_libs)}")
    # Keep only libraries that are actually available
    selected_gene_set_libraries = [
        'KEGG_2021_Human',
        # 'GTEx_Tissue_Expression_Down', # Often less useful than combined/V8
        # 'GTEx_Tissue_Expression_Up',   # Often less useful than combined/V8
        'GTEx_Tissue_V8_Expression',
        'OMIM_Disease',
        "OMIM_Expanded",
        # 'Panther_2016', # Quite old, consider newer pathway DBs if available
        'Reactome_2022',
        'TRANSFAC_and_JASPAR_PWMs',
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023',
        'MSigDB_Hallmark_2020',
    ]
    # Filter list based on availability
    valid_libraries = [lib for lib in selected_gene_set_libraries if lib in available_libs]
    missing_libs = [lib for lib in selected_gene_set_libraries if lib not in available_libs]
    if missing_libs:
        logging.warning(f"Libraries not found in enrichrpy's list and will be skipped: {missing_libs}")
        # Optional: Log some available libraries if many are missing
        # logging.info(f"Some available libraries: {available_libs[:20]}") # Show first 20
    selected_gene_set_libraries = valid_libraries
    logging.info(f"Using valid libraries for analysis: {selected_gene_set_libraries}")

except Exception as e:
    logging.error(f"Failed to retrieve available libraries from Enrichr: {e}. Falling back to default list.", exc_info=True)
    # Fallback to the original list if the check fails, but processing might fail later
    selected_gene_set_libraries = [
        'KEGG_2021_Human', 'GTEx_Tissue_V8_Expression', 'OMIM_Disease',
        "OMIM_Expanded", 'Reactome_2022', 'TRANSFAC_and_JASPAR_PWMs',
        'GO_Biological_Process_2023', 'GO_Molecular_Function_2023',
        'MSigDB_Hallmark_2020',
    ]
    logging.warning(f"Proceeding with fallback library list: {selected_gene_set_libraries}")


def find_pathways_enrichrpy(sample_genes, libraries_to_query):
    """
    Finds pathways using enrichrpy, querying multiple libraries.

    Args:
        sample_genes (list): A list of gene symbols.
        libraries_to_query (list): A list of Enrichr library names to query.

    Returns:
        pandas.DataFrame: DataFrame containing top 10 enrichment results per library,
                          or empty DataFrame on failure or if no results found.
    """
    # Validate input
    if not isinstance(sample_genes, list) or not sample_genes:
        logging.error("Input 'sample_genes' must be a non-empty list.")
        return pd.DataFrame()
    # Filter out potential non-string or empty string elements
    sample_genes = [str(g) for g in sample_genes if isinstance(g, str) and g.strip()]
    if not sample_genes:
        logging.error("Input 'sample_genes' contained no valid gene symbols after cleaning.")
        return pd.DataFrame()
    if not isinstance(libraries_to_query, list) or not libraries_to_query:
        logging.error("Input 'libraries_to_query' must be a non-empty list.")
        return pd.DataFrame()

    logging.info(f"Starting enrichment analysis for {len(sample_genes)} genes across {len(libraries_to_query)} libraries using enrichrpy.")

    all_results_dfs = [] # To store DataFrames from each library

    for gene_set_library in libraries_to_query:
        logging.info(f"--- Processing library: {gene_set_library} ---")
        try:
            # Call enrichrpy to get enrichment results for the current library
            # enrichrpy handles the API calls internally
            enr_df = een.get_pathway_enrichment(
                genes=sample_genes,
                gene_set_library=gene_set_library
            )

            # Check if results were returned
            if enr_df is None or enr_df.empty:
                logging.info(f"  No enrichment results found for library: {gene_set_library}")
                continue # Skip to the next library

            logging.info(f"  Successfully retrieved {len(enr_df)} results for {gene_set_library}.")

            # --- Data Processing and Selection ---
            # Add the library name as a column
            enr_df["gene_set_library"] = gene_set_library

            # Check essential columns returned by enrichrpy (names might vary slightly)
            # Common columns: 'Term', 'P-value', 'Adjusted P-value', 'Combined Score', 'Genes'
            # Rename columns to match generate_prompt's expected input
            # Adjust these mappings based on actual columns returned by enrichrpy if needed
            column_mapping = {
                "Term": "description",
                "P-value": "p_value",      # Ensure case matches output
                "Combined Score": "combined_score", # Ensure case matches output
                "Genes": "genes"           # Ensure case matches output
                # Add other renames if necessary
            }

            # Check which columns actually exist before renaming
            rename_map_filtered = {k: v for k, v in column_mapping.items() if k in enr_df.columns}
            enr_df = enr_df.rename(columns=rename_map_filtered)

            # Verify required columns for filtering and generate_prompt are present after renaming
            required_cols = ['description', 'p_value', 'combined_score', 'genes', 'gene_set_library']
            missing_req_cols = [col for col in required_cols if col not in enr_df.columns]
            if missing_req_cols:
                 logging.warning(f"  Skipping library {gene_set_library} due to missing required columns after rename: {missing_req_cols}. Available columns: {enr_df.columns.tolist()}")
                 continue

            # Ensure p_value is numeric for sorting/filtering
            enr_df['p_value'] = pd.to_numeric(enr_df['p_value'], errors='coerce')
            enr_df = enr_df.dropna(subset=['p_value']) # Remove rows where p_value couldn't be parsed

            if enr_df.empty:
                logging.info(f"  No valid results remained for {gene_set_library} after cleaning/type conversion.")
                continue

            # Filter top 10 by p-value (lowest p-values are best)
            # enrichrpy usually returns sorted results, but explicitly sorting/filtering is safer
            enr_df_top10 = enr_df.nsmallest(10, 'p_value')

            # Select only the columns needed downstream
            final_cols = required_cols # Use the list defined above
            # Filter final_cols to only those that actually exist in enr_df_top10
            final_cols = [col for col in final_cols if col in enr_df_top10.columns]
            res_df_filtered = enr_df_top10[final_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

            # --- Ensure 'genes' column is a list of strings ---
            # enrichrpy usually returns a list directly from the API JSON.
            # If it's a semicolon-separated string, uncomment the next line:
            # res_df_filtered['genes'] = res_df_filtered['genes'].apply(lambda x: x.split(';') if isinstance(x, str) else x)
            # Let's add a check to be sure
            if not res_df_filtered.empty and not isinstance(res_df_filtered['genes'].iloc[0], list):
                 logging.warning(f"  'genes' column in {gene_set_library} doesn't appear to be a list. Type: {type(res_df_filtered['genes'].iloc[0])}. Trying to split if string.")
                 # Attempt split only if it's a string and contains semicolons typical of Enrichr output
                 if isinstance(res_df_filtered['genes'].iloc[0], str) and ';' in res_df_filtered['genes'].iloc[0]:
                      try:
                           res_df_filtered['genes'] = res_df_filtered['genes'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
                      except Exception as split_err:
                           logging.error(f"   Failed to split 'genes' column string for {gene_set_library}: {split_err}")
                           # Decide how to handle: keep as is, set to empty list, or skip? Let's keep as is for now.

            all_results_dfs.append(res_df_filtered)
            logging.info(f"  Added top {len(res_df_filtered)} results from {gene_set_library} to final list.")

            # Optional: Add a small delay between library requests if you suspect rate limiting
            # time.sleep(0.5) # e.g., wait 0.5 seconds

        except Exception as e:
            # Catch potential errors during the enrichrpy call or processing
            logging.error(f"  Failed to process library {gene_set_library}: {e}", exc_info=True)
            # Continue to the next library

    # --- Combine results ---
    if not all_results_dfs:
        logging.warning("No enrichment results found for any library.")
        return pd.DataFrame()

    try:
        results = pd.concat(all_results_dfs, ignore_index=True)
        logging.info(f"--- Enrichment analysis finished. Returning {len(results)} total results from {len(all_results_dfs)} libraries. ---")
        return results
    except Exception as e:
        logging.error(f"Failed to concatenate results: {e}", exc_info=True)
        return pd.DataFrame() # Return empty if concatenation fails


# --- Keep your generate_prompt function as is ---
# (Make sure it handles the column names correctly as prepared above)
def generate_prompt(results, gene_list):
    """Generate a prompt for the LLM based on aggregated results."""
    prompt = f"For this list of genes: {', '.join(gene_list)}, the relevant findings from Enrichr are:\n"

    if results is None or results.empty:
        prompt += "\nNo relevant enrichment findings were identified or the analysis failed to return results."
        return prompt

    library_summary = {}

    required_cols = ['gene_set_library', 'description', 'p_value', 'combined_score', 'genes']
    if not all(col in results.columns for col in required_cols):
        logging.error("generate_prompt: Results DataFrame is missing required columns.")
        prompt += "\nError processing Enrichr results due to missing data columns."
        return prompt

    # Ensure sorting columns are numeric where expected
    results['p_value'] = pd.to_numeric(results['p_value'], errors='coerce')
    results['combined_score'] = pd.to_numeric(results['combined_score'], errors='coerce')
    results = results.dropna(subset=['p_value']) # Drop rows if p_value became NaN

    if results.empty:
         prompt += "\nNo valid enrichment results remained after cleaning."
         return prompt

    # Sort overall results primarily by library, then by p-value
    results_sorted = results.sort_values(by=['gene_set_library', 'p_value'], ascending=[True, True])

    prompt += "\n--- Top Enrichment Results Summary (up to 10 per library) ---\n"
    for library, group in results_sorted.groupby('gene_set_library'):
        if not group.empty:
            # Group is already sorted by p-value because results_sorted was sorted
            top_entry = group.iloc[0]
            # Get unique descriptions from the top N entries for this library (already filtered to <=10)
            descriptions = list(group['description'].unique())
            # Format genes nicely
            genes_display = top_entry['genes']
            if isinstance(genes_display, list):
                genes_str = ', '.join(genes_display)
            elif genes_display is None or pd.isna(genes_display):
                 genes_str = "N/A"
            else:
                 genes_str = str(genes_display) # Fallback

            prompt += (f"\n* Library: {library}\n"
                       f"  Top Term: '{top_entry['description']}'\n"
                       f"    P-Value: {top_entry['p_value']:.3e}\n" # Scientific notation for p-value
                       f"    Combined Score: {top_entry['combined_score']:.4f}\n"
                       f"    Overlapping Genes (in top term): {genes_str}\n"
                       f"  Other Top Terms in this library: {'; '.join(descriptions[1:]) if len(descriptions) > 1 else 'None'}\n")

    prompt += "\n--- End of Summary ---\n"
    return prompt


# --- Example Usage ---
if __name__ == "__main__":
    genes = [
        'TYROBP', 'HLA-DRA', 'SPP1', 'LAPTM5', 'C1QB',
        'FCER1G', 'GPNMB', 'FCGR3A', 'RGS1', 'HLA-DPA1',
        'ITGB2', 'C1QC', 'HLA-DPB1', 'IFI30', 'SRGN',
        'APOC1', 'CD68', 'HLA-DRB1', 'C1QA', 'LYZ',
        'APOE', 'HLA-DQB1', 'CTSB', 'HLA-DQA1', 'CD74',
        'AIF1', 'FCGR2A', 'CD14', 'S100A9', 'CTSS'
    ]

    # Use the validated list of libraries
    enrichment_results = find_pathways_enrichrpy(genes, selected_gene_set_libraries)

    if not enrichment_results.empty:
        print("\n--- Enrichment Results DataFrame (Head) ---")
        print(enrichment_results.head())

        # Check the data types of the relevant columns
        print("\n--- DataFrame Info ---")
        enrichment_results.info()

        # Generate the prompt for the LLM
        llm_prompt = generate_prompt(enrichment_results, genes)
        print("\n--- Generated LLM Prompt ---")
        print(llm_prompt)

        # --- Optional: Generate plots using enrichrpy.plotting ---
        # Note: Plotting usually works best on results from a SINGLE library.
        # You might want to filter the results before plotting.
        try:
            import enrichrpy.plotting as epl
            import altair as alt
            # Example: Plot for the first library found in the results
            first_library = enrichment_results['gene_set_library'].iloc[0]
            df_for_plot = enrichment_results[enrichment_results['gene_set_library'] == first_library]

            # Check if Z-score exists for dotplot hue
            if 'Z-score' in df_for_plot.columns:
                # Convert Z-score to numeric if needed
                df_for_plot['Z-score'] = pd.to_numeric(df_for_plot.get('Z-score'), errors='coerce') # Use .get for safety
                # Make sure hue column does not contain NaNs if used
                df_plot_ready = df_for_plot.dropna(subset=['Z-score'])
                if not df_plot_ready.empty:
                    print(f"\n--- Generating Dot Plot for library: {first_library} ---")
                    dotplot = epl.enrichment_dotplot(df_plot_ready, n=10, hue='Z-score', log=True, title=f"Top Terms for {first_library}")
                    # To view/save the plot:
                    dotplot.show() # Opens in browser/viewer
                    # dotplot.save(f"{first_library}_dotplot.png") # Requires altair_saver and dependencies
                    # dotplot.save(f"{first_library}_dotplot.svg")
                else:
                     print(f"\n--- Cannot generate Dot Plot for {first_library} (missing Z-score data after cleaning) ---")

            else:
                 # Generate bar plot if Z-score is missing
                 print(f"\n--- Generating Bar Plot for library: {first_library} ---")
                 barplot = epl.enrichment_barplot(df_for_plot, n=10, title=f"Top Terms for {first_library}")
                 barplot.show()
                 # barplot.save(f"{first_library}_barplot.png")

        except ImportError:
            logging.warning("Plotting libraries (enrichrpy.plotting, altair) not installed or import failed. Skipping plot generation.")
        except Exception as plot_err:
            logging.error(f"Error during plot generation: {plot_err}", exc_info=True)

    else:
        print("No enrichment results were generated.")