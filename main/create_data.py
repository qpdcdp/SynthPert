import os
import argparse

# --- Set up argparse for command line arguments ---

#TODO: add majority voting

parser = argparse.ArgumentParser(description="Create synthetic data.")

def comma_separated_list(value):
    if ',' in value:
        return [item.strip() for item in value.split(',')]
    return [value.strip()]

parser.add_argument(
    "--output_dir",
    type=str,
    default="./output/synth_data/",
    help="Directory to save the output files.",
)
parser.add_argument(
    "--generator_model_name",
    type=str,
    default="openai_o3",
    help="Name of the generator model to use.",
)
parser.add_argument(
    "--critic_model_name",
    type=str,
    default="openai_o3",
    help="Name of the critic model to use.",
)
parser.add_argument(
    "--train_subset_fraction",
    type=float,
    default=0.15,
    help="Fraction of the training set to use.",
)
parser.add_argument(
    "--task",
    type=str,
    default="single_gene_prediction",
    choices=["single_gene_prediction", "direct_prediction"],
    help="Task type for training.",
)
parser.add_argument(
    "--critic_lemmas",
    action="store_true",
    help="Lemmas list for the critic model.",
)
parser.add_argument(
    "--generator_lemmas",
    action="store_true",
    help="Lemmas list for the generator model.",
)
parser.add_argument(
    "--context",
    type=str,
    default="none",
    help="Context to use for the synthetic data.",
)
parser.add_argument(
    "--tool",
    type=str,
    default="none",
    help="Allow model to call external gene enrichment libraries.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=40,
    help="Number of workers for api calls.",
)
parser.add_argument(
    "--synth_data_script",
    type=str,
    default="with_critic",
)
parser.add_argument(
    "--csv_data_directory",
    type=str,
    default="./data",
    help="Directory containing the CSV data files.",
)
parser.add_argument(
    "--test_split_cell_lines",
    type=str,
    default="none",
    help="Cell lines to use for the test split.",
)
parser.add_argument(
    "--critic_acceptance_threshold",
    type=comma_separated_list,
    help="Acceptance threshold for the critic model.",
    default="excellent,good",
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

    print("Generating data with critic")
    from src.generate_data.create_synth_data_with_critic import main
    main(args)
