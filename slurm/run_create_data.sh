#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -c 48

# === Define default values using positional parameter expansion ===
#SBATCH --job-name=create_synth_data_without_critic
#SBATCH --output=logs/create_data/%x_%j.log
#SBATCH --error=logs/create_data/%x_%j.err


# Define the create data args parse variables
export MODEL_NAME="openai_o4_mini"
export SYNTH_SCRIPT="without_critic"
export TOOL="none"
export TEST_SET_CELL_LINES="default"
export TRAIN_DATA_FRACTION=0.15
export CRITIC_ACCEPTANCE_THRESHOLD="excellent,good"
export MARKETPLACE_URL="your langchain model marketplace url"
export API_KEY="your langchain marketplace api key"

find_empty_port() {
    for port in {1024..65535}; do
        if ! nc -z localhost $port 2>/dev/null; then
            echo $port
            return 0
        fi
    done
    echo "no available port found" >&2
    return 1
}


export MASTER_PORT="$(find_empty_port)"
export MASTER_NAME="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "$MASTER_NAME" hostname --ip-address)"

echo "The head node name is $MASTER_NAME"
echo "The head node IP is $MASTER_ADDR"


# Define base path clearly so it's accessible inside the bash command
BASE_PATH="/novo/projects/departments/mi/lwph/PertRL"

# Use srun to properly utilize allocated resources
srun \
    --export=ALL \
    bash -c "export PYTHONPATH=${BASE_PATH}:\$PYTHONPATH && \
    export PYTHONUNBUFFERED=1 && \
    source \$(conda info --base)/etc/profile.d/conda.sh && \
    conda activate novopertrl && \
    cd ${BASE_PATH} && \
    accelerate launch \
    --num_cpu_threads_per_process=8 \
    --config_file=${BASE_PATH}/src/configs/api_eval.yaml \
    ${BASE_PATH}/main/create_data.py \
    --synth_data_script \"${SYNTH_SCRIPT}\" \
    --generator_model_name \"${MODEL_NAME}\" \
    --critic_model_name \"${MODEL_NAME}\" \
    --tool \"${TOOL}\" \
    --test_split_cell_lines \"${TEST_SET_CELL_LINES}\" \
    --train_subset_fraction \"${TRAIN_DATA_FRACTION}\" \
    --generator_lemmas \
    --marketplace_api_key \"${API_KEY}\" \
    --marketplace_url \"${MARKETPLACE_URL}\" \
    --critic_acceptance_threshold \"${CRITIC_ACCEPTANCE_THRESHOLD}\""