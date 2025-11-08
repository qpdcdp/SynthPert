import os
import argparse

# --- Set up argparse for command line arguments ---
# pass these through another file if needed

parser = argparse.ArgumentParser(description="Train a model with LoRA.")
parser.add_argument(
    "--train_stage", 
    type=str, 
    default="SFT", 
    help="Stage of training: SFT or GRPO")

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for initialization",
)
parser.add_argument(
    "--lora",
    action="store_true",
    help="Whether to use LoRA for training.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    help="Path to the pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output/SFT/SFT_lora_all_cell_lines",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(    
    "--train_batch_size",
    type=int,
    default=4,
    help="Batch size per device during training.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=2,
    help="Batch size for evaluation.",
)
parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=96,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate for Adam.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--test_split_cell_lines",
    type=str,
    default="none",
    help="Cell lines to use for the test split.",
)
parser.add_argument(
    "--task",
    type=str,
    default="on_data",
    choices=["COT", "on_data", "direct_prediction", "gene_regulation", "metabolic_flux", "seahorse_cluster", "inverse_problem"],
    help="Task type for training.",
)
parser.add_argument(
    "--synth_data_csv",
    type=str,
    default="./output/synth_data/synthetic_data_openai_o4_mini_with_critic_default_split_excellent_only.csv",
    help="Path to the synthetic data CSV file.",
)
parser.add_argument(
    "--list_format",
    type=str,
    default="default",
    help="Format of the gene list in the output.",
)
#GRPO specific
parser.add_argument(
    "--lora_checkpoint_path",
    type=str,
    default="./output/SFT/SFT_lora/checkpoint-1206",
    help="Path to the SFT checkpoint to load.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train_stage == "SFT":
        if args.lora:
            if args.task == "on_data":
                print("Using on_data train")
                from src.train.sft_train_hf_lora_on_data import main
                main(args)
            else:
                print("Using COT train")
                from src.train.sft_train_hf_lora import main
                main(args)
        else:
            from src.train.sft_full_hf import main
            main(args)
    elif args.train_stage == "GRPO":
        if args.lora:
            from src.train.grpo_lora_train import main
            main(args)
        else:
            from src.train.grpo_train import main
            main(args)