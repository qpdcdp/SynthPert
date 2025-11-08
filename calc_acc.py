import json
import glob
from sklearn.metrics import classification_report

def parse_results_from_file(file_path: str) -> tuple[list, list]:
    """
    Parses a single JSONL file and returns lists of true and predicted values.

    This function reads a given file and extracts the 'correct_solution' and
    'extracted_answer' pairs, handling errors gracefully.

    Args:
        file_path (str): The path to the input JSONL file.

    Returns:
        A tuple containing two lists: (y_true, y_pred).
    """
    y_true = []
    y_pred = []

    print(f"INFO: Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    correct_sol = data.get('correct_solution')
                    extracted_ans = data.get('extracted_answer')

                    if correct_sol is not None and extracted_ans is not None:
                        y_true.append(correct_sol)
                        y_pred.append(extracted_ans)
                    else:
                        print(f"Warning: Skipping line {line_num} in {file_path} due to missing or null values.")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {line_num} in {file_path}.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        # Return empty lists to allow the main loop to continue
        return [], []

    return y_true, y_pred

def main():
    """
    Finds all 'temp_results_rank_*.jsonl' files in a directory, aggregates
    their data, and prints a single, combined classification report.
    """
    # --- MODIFIED SECTION START ---

    # 1. Define the directory and the pattern for the files you want to process
    eval_directory = "/novo/projects/departments/mi/lwph/CellPert/results/single_gene_prediction/rpe1_specific_genes/on_data/llama-8b/checkpoint-10000/"
    file_pattern = "eval.jsonl"

    # 2. Use glob to find all files matching the pattern

    # if not file_paths:
    #     print(f"Error: No files found matching the pattern '{file_pattern}'.")
    #     print("Please check the directory and pattern.")
    #     return

    # 3. Initialize master lists to aggregate data from all files
    all_y_true = []
    all_y_pred = []

    # 4. Loop through each found file, parse it, and extend the master lists
    for path in ["/novo/projects/departments/mi/lwph/CellPert/results/single_gene_prediction/rpe1_specific_genes/on_data/llama-8b/checkpoint-10000/eval.jsonl"]:
        y_true, y_pred = parse_results_from_file(path)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # --- MODIFIED SECTION END ---

    if not all_y_true:
        print("\nNo valid entries were found across all files. Cannot generate report.")
        return

    # 5. Generate and print the single, combined classification report
    print("\n--- Combined Classification Report ---")
    # print(f"Aggregated from {len(file_paths)} files.")
    print("------------------------------------")

    report = classification_report(all_y_true, all_y_pred, zero_division=0)
    print(report)
    
    print("------------------------------------\n")
    print("Glossary:")
    print("  - precision: Of all predictions for a class, how many were correct? (TP / (TP + FP))")
    print("  - recall:    Of all true instances of a class, how many were correctly predicted? (TP / (TP + FN))")
    print("  - f1-score:  The harmonic mean of precision and recall.")
    print("  - support:   The number of actual occurrences of the class in the dataset.")
    print("  - macro avg: Average of metrics, giving equal weight to each class.")
    print("  - weighted avg: Average of metrics, weighted by the support of each class.")


if __name__ == "__main__":
    main()