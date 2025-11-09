import pandas as pd
import json
import requests
import tempfile
import os

def find_pathways(sample_genes):
    # Write the sample genes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_gene_file:
        temp_gene_file.write("\n".join(sample_genes))
        gene_file_path = temp_gene_file.name
        
    # Enrichr API URLs
    ENRICHR_ADD = 'http://amp.pharm.mssm.edu/Enrichr/addList'
    ENRICHR_RETRIEVE = 'http://amp.pharm.mssm.edu/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'
    
    # Selected gene set libraries
    selected_gene_set_libraries = [
        'KEGG_2021_Human',
        # 'GTEx_Tissue_Expression_Down', # These seem less standard, replaced by tissue-specific below
        # 'GTEx_Tissue_Expression_Up',
        'GTEx_Tissue_V8_Expression', # Broad GTEx V8 expression
        # 'GTEx_Tissues_V8_2023', # Name might be slightly off, using above
        'OMIM_Disease',
        "OMIM_Expanded",
        'Panther_2016', # Might be old, but often available
        'Reactome_2022', # Use dated version which is more likely stable
        'TRANSFAC_and_JASPAR_PWMs',
        # 'GO_Biological_Process_2023', # Added popular GO library
        # 'GO_Molecular_Function_2023',
        'MSigDB_Hallmark_2020',  
    ]
    
    # Read the gene names from the temporary file
    gene_names = pd.read_csv(gene_file_path, header=None).squeeze().tolist()
    
    # Print the number of genes
    #print("Gene set has %i genes" % len(gene_names))
    
    # Container for storing the results
    results = pd.DataFrame()
    
    for gene_set_library in selected_gene_set_libraries:
        #print(f"Running Enrichr on {gene_set_library} gene set library.")
        
        # Build payload with gene list
        attr = "\n".join(gene_names)
        payload = {
            'list': (None, attr),
            'description': (None, gene_set_library)
        }
        try:
            # Request adding gene set
            response = requests.post(ENRICHR_ADD, files=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error analyzing gene list for {gene_set_library}: {e}")
            continue
        
        # Try tracking gene set ID
        try:
            user_list_id = json.loads(response.text)['userListId']
        except KeyError:
            print(f"Unexpected response data for {gene_set_library}: {response.text}")
            continue
        
        try:
            # Request enriched sets in gene set
            response = requests.get(ENRICHR_RETRIEVE + query_string % (user_list_id, gene_set_library))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching enrichment results for {gene_set_library}: {e}")
            continue
        
        # Parse response JSON
        res = json.loads(response.text)
        
        # Check if the response has the expected keys
        if gene_set_library in res and isinstance(res[gene_set_library], list):
            enrich_data = res[gene_set_library]
            if len(enrich_data) > 0:
                # Convert the data to a dataframe
                res_df = pd.DataFrame.from_records(enrich_data)
                # Set column names
                res_df.columns = ["rank", "description", "p_value", "z_score", "combined_score", "genes", "adjusted_p_value", "unknown_col1", "unknown_col2"]
                # Add gene set library used
                res_df["gene_set_library"] = gene_set_library
                
                # Filter to include only the top 10 by p-value
                res_df = res_df.nsmallest(10, 'p_value')
                
                # Use pd.concat to append the dataframe
                results = pd.concat([results, res_df], ignore_index=True)
        else:
            print(f"No enrichment data found for {gene_set_library}")
    
    # Clean up the temporary gene file
    os.remove(gene_file_path)
    
    # Return results
    return results

def generate_prompt(results, gene_list):
    """Generate a prompt for the LLM based on aggregated results."""
    prompt = f"For this list of genes: {', '.join(gene_list)}, the relevant findings are:\n"
    
    if not results.empty:
        library_summary = {}
        
        # Aggregate results by gene set library
        for index, row in results.iterrows():
            library = row['gene_set_library']
            if library not in library_summary:
                library_summary[library] = {
                    "descriptions": [],
                    "top_score": row['p_value'],
                    "top_combined_score": row['combined_score'],
                    "top_genes": row['genes']
                }
            # Track description, if it's a new best result
            library_summary[library]["descriptions"].append(row['description'])
            
            # Check for top scoring entries
            if row['p_value'] < library_summary[library]["top_score"]:
                library_summary[library]["top_score"] = row['p_value']
                library_summary[library]["top_combined_score"] = row['combined_score']
                library_summary[library]["top_genes"] = row['genes']
        
        # Create prompt output from the aggregated findings
        for library, info in library_summary.items():
            prompt += (f"- Gene Set Library: {library}\n"
                       f"  Top P-Value: {info['top_score']}\n"
                       f"  Top Combined Score: {info['top_combined_score']}\n"
                       f"  Top Genes: {info['top_genes']}\n"
                       f"  Descriptions: {', '.join(set(info['descriptions']))}\n\n")
    else:
        prompt += "No relevant findings were identified."
    
    return prompt

# Example usage:
if __name__ == "__main__":
    sample_genes = [
'EEF1A1', 'HSPE1', 'RPL13', 'RPL23', 'RPL23A', 'RPL27A', 'RPL3', 'RPL37', 'RPS14', 'RPS16', 'RPS23', 'RPS25', 'RPS28', 'RPS3', 'RPS8', 'RPS9', 'TPT1'
    ]
    results = find_pathways(sample_genes)
    prompt = generate_prompt(results, sample_genes)

    # Save the results to a CSV file
    #output_file = "sample_enrichr_output_top_10.csv"
    #results.to_csv(output_file, index=False, encoding='utf-8')
    #print(f"Results are saved to {output_file}")

    # Print or save the prompt
    print(prompt)