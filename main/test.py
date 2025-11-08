import os
import argparse

# --- Set up argparse for command line arguments ---
# pass these through another file if needed

parser = argparse.ArgumentParser(description="Test model on test set.")
parser.add_argument(
    "--test_script",
    type=str,
    default="hf",
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
    default="None",
    help="Tool use for testing.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
)
parser.add_argument(
    "--task",
    type=str,
    default="single_gene_prediction",
    help="Task to perform.",
)
parser.add_argument(
    "--list_format",
    type=str,
    default="default",
    help="Format of the gene list in the output.",
)
parser.add_argument(
    "--partial_list_fraction",
    type=float,
    default=0.0,
    help="faction of ground truth regulated lists provided to model"
)
parser.add_argument(
    "--dataset_fraction",
    type=float,
    default=1.0,
    help="fraction of the dataset to use for testing",
)
parser.add_argument(
    "--warning_different_distributions",
    action="store_true",
    help="Whether to warn the LLM about different distributions in the dataset in the User Query.",
)
parser.add_argument(
    #change this to generate_all_non_de_samples
    "--generate_all_non_de_samples",
    action="store_true",
    help="Whether to use all genes in file as not differentially expressed.",
)
parser.add_argument(
    "--generate_4x_non_de_samples",
    action="store_true",
    help="Whether to generate 4x non-DE samples.",
)
parser.add_argument(
    "--gene_enrichment",
    action="store_true",
    help="Whether to perform gene enrichment analysis.",
)
parser.add_argument(
    "--eval_unique_genes_to_test_only",
    action="store_true",
    help="Whether to ensure that test set perturbations are unique and not present in the training set.",
)
parser.add_argument(
    "--num_folds", 
    type=int, 
    default=10, 
    help="Number of folds to split the data into for error estimation. If 1, runs a single standard evaluation without error bars.",
)
parser.add_argument(
    "--error_estimation",
    action="store_true",
    help="Whether to perform error estimation.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--devices",
    type=int,
    default=1,
    help="Number of devices to use for testing.",
)
parser.add_argument(
    "--num_nodes",
    type=int,
    default=1,
    help="Number of nodes to use for testing.",
)
parser.add_argument(
    "--strategy",
    type=str,
    default="auto",
    help="Strategy for distributed testing.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1,
    help="Temperature for sampling during generation.",
)
parser.add_argument(
    "--test_mode",
    type=str,
    default="single_gene_prediction",
    help="Testing mode to use.",
)

parser.add_argument(
    "--vllm_gpu_memory_utilization",
    type=float,
    default=0.9,  # vLLM's default is 0.9, we'll allow overriding it
    help="The fraction of GPU memory to be used for the vLLM KV cache."
)

parser.add_argument(
    "--vllm_max_num_seqs",
    type=int,
    default=256, # A common default, we will override this in the SLURM script
    help="Maximum number of sequences for the vLLM engine."
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.test_script == "fabric":
        from src.test.test_single_gene_prediction import main
        print("Starting test_single_gene_prediction.py")
        main(args)
    elif args.test_script == "vllm":
        from src.test.test_single_gene_prediction_vllm import main
        print("Starting test_single_gene_prediction_vllm.py")
        main(args)
    elif args.test_script == "hf" and args.task == "single_gene_prediction" and args.error_estimation:
        from src.test.test_tuples_model_with_errors import main
        print("Starting test_tuples_model_with_errors.py")
        main(args)
    elif args.test_script == "hf" and args.task == "single_gene_prediction":
        from src.test.test_tuples_model import main
        print("Starting test_tuples_model.py")
        main(args)
    elif args.test_script == "hf" and args.task == "direct_prediction":
        from src.test.test_hf_model import main
        print("Starting test_hf_model.py")
        main(args)
    elif args.test_script == "api":
        from src.test.test_api_model import main
        main(args)
