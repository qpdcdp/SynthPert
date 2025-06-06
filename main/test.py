import os
import argparse

# --- Set up argparse for command line arguments ---
# pass these through another file if needed

parser = argparse.ArgumentParser(description="Test model on test set.")
parser.add_argument(
    "--test_script",
    type=str,
    default="api",
    help="Script to run for testing the model.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output/eval/hf_model/new_model/",
    help="Directory to save the output files.",
)
parser.add_argument(
    "--tool",
    type=str,
    default="enrichr",
    help="Tool use for testing.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="DeepSeek-R1-Distill-Llama-8B",
    help="Path to the pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for testing.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=4096,
    help="Max number of tokens for generation.",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to the checkpoint to load.",
)
parser.add_argument(
    "--lora_checkpoint",
    type=str,
    default="./output/test",
    help="Path to the lora checkpoint to load.",
)
parser.add_argument(
    "--test_split_cell_lines",
    type=str,
    default="none",
    help="Cell lines to use for the test split.",
)
parser.add_argument(
    "--AUROC",
    action="store_true",
    help="Whether to use AUROC for evaluation.",
)
parser.add_argument(
    "--AUROC_stage",
    type=str,
    default="dif",
    help="Stage of AUROC evaluation.",
)
parser.add_argument(
    "--csv_data_directory",
    type=str,
    default="./data",
    help="Directory containing the CSV data files.",
)
parser.add_argument(
    "--task",
    type=str,
    default="single_gene_prediction",
    choices=["single_gene_prediction", "direct_prediction", "triplets"],
    help="Task type for training.",
)
parser.add_argument(
    "--save_interval_batches",
    type=int,
    default=100,  # Default to 0 (disabled)
    help="Save results periodically every N batches. If 0, only saves at the end."
)
parser.add_argument(
    "--marketplace_url",
    type=str,
    help="url to your langchain model marketplace",
)
parser.add_argument(
    "--marketplace_api_key",
    type=str,
    help="langchain marketplace api key"
)
if __name__ == "__main__":
    args = parser.parse_args()
    if args.test_script == "api":
        from src.test.test_api import main
        main(args)
    else:
        if args.AUROC:
            print("AUROC evaluation")
            from src.test.test_hf_model_AUROC import main
            main(args)
        else:
            from src.test.test_hf_model_base import main
            main(args)
