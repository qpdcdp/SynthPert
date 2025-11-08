import json
import re
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def parse_prediction_from_answer_tag(prediction_str: str) -> str | None:
    """
    Extracts the predicted label from the <answer> tag in a given string.

    Args:
        prediction_str: The string containing the prediction, typically with XML-like tags.

    Returns:
        The cleaned prediction string (e.g., 'upregulated') or None if not found.
    """
    # Use a non-greedy regex to find content within the first <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', prediction_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_jsonl_file(filepath: str) -> tuple[list, list]:
    """
    Reads a JSONL file, parses each line as a JSON object, and extracts
    the true and predicted labels from all perturbations.

    Args:
        filepath: The path to the JSONL file.

    Returns:
        A tuple containing two lists: y_true (ground truth labels) and
        y_pred (predicted labels).
    """
    y_true = []
    y_pred = []
    total_lines = 0
    parsed_lines = 0

    print(f"Processing file: {filepath}...")

    with open(filepath, 'r') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                # The predictions are in a list under the "initial_predictions" key
                predictions = data.get("initial_predictions", [])

                for item in predictions:
                    true_label = item.get("ground_truth_label")
                    prediction_str = item.get("prediction")

                    if true_label and prediction_str:
                        predicted_label = parse_prediction_from_answer_tag(prediction_str)
                        if predicted_label:
                            y_true.append(true_label)
                            y_pred.append(predicted_label)
                        else:
                            # Log a warning if a prediction can't be parsed
                            gene = item.get('gene', 'Unknown')
                            print(f"  [Warning] Could not parse prediction for gene '{gene}' in perturbation '{data.get('perturbation', 'Unknown')}'")
                
                parsed_lines += 1

            except json.JSONDecodeError as e:
                print(f"  [Error] Skipping malformed JSON line #{total_lines}: {e}")
                continue
            except Exception as e:
                print(f"  [Error] An unexpected error occurred on line #{total_lines}: {e}")

    print(f"Finished processing. Parsed {parsed_lines}/{total_lines} lines.")
    return y_true, y_pred

def generate_full_report(y_true: list, y_pred: list):
    """
    Generates and prints a full classification report including accuracy,
    precision, recall, f1-score, and a confusion matrix.

    Args:
        y_true: A list of ground truth labels.
        y_pred: A list of predicted labels.
    """
    if not y_true or not y_pred:
        print("No valid predictions found to generate a report.")
        return

    # Get all unique labels present in either true or predicted sets
    labels = sorted(list(set(y_true + y_pred)))
    
    print("\n" + "=" * 60)
    print("           Full Aggregate Classification Report")
    print("=" * 60)
    
    # 1. Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print(f"Total Predictions Evaluated: {len(y_true)}\n")

    # 2. Classification Report (Precision, Recall, F1-Score)
    print("--- Classification Report ---")
    # Use zero_division=0 to avoid warnings if a class has no predicted samples
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    
    # 3. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Use pandas for a more readable, labeled table format
    cm_df = pd.DataFrame(cm, index=[f'Actual: {l}' for l in labels], columns=[f'Predicted: {l}' for l in labels])
    print(cm_df)
    print("\n(Rows are actual classes, Columns are predicted classes)")
    print("=" * 60)


if __name__ == "__main__":
    # Set up argument parser to accept a file path from the command line
    parser = argparse.ArgumentParser(
        description="Generate a full classification report from a JSONL file of gene expression predictions."
    )
    parser.add_argument(
        "--filepath", 
        help="Path to the input JSONL file."
    )

    args = parser.parse_args()

    try:
        # Process the file to get true and predicted labels
        true_labels, pred_labels = process_jsonl_file(args.filepath)
        
        # Generate and print the comprehensive report
        generate_full_report(true_labels, pred_labels)
        
    except FileNotFoundError:
        print(f"[Error] The file '{args.filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")