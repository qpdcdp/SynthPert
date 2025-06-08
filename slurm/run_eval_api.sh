#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --time=24:00:00


export TOOL="None" 
export BATCH_SIZE=64
export DATA_SPLIT="rpe1" #type of split, rpe1 is for table 5, "default" for others (default train/test split)
export STAGE="dif"
export MARKETPLACE_URL="your langchain model marketplace url"
export API_KEY="your langchain marketplace api key"
export MODEL = "openai_o4_mini"
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
export HF_TOKEN="{your HuggingFace token}"

export MASTER_PORT="$(find_empty_port)"
export MASTER_NAME="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "$MASTER_NAME" hostname --ip-address)"


BASE_PATH="/path/to/project_dir"


# Use srun to properly utilize allocated resources
srun \
    --export=ALL \
    bash -c "export PYTHONPATH=${BASE_PATH}:\$PYTHONPATH && \
    export PYTHONUNBUFFERED=1 && \
    source \$(conda info --base)/etc/profile.d/conda.sh && \
    conda activate marc_unsloth && \
    cd ${BASE_PATH} && \
    export PYTHONPATH=${BASE_PATH} && \ 
    python ${BASE_PATH}/main/test.py \
    --AUROC_stage ${STAGE} \
    --test_script "api" \
    --tool "${TOOL}" \
    --test_split_cell_lines "${DATA_SPLIT}" \
    --batch_size ${BATCH_SIZE} \
    --model_name "${MODEL}" \ 
    --marketplace_api_key "${API_KEY}\" \
    --marketplace_url "${MARKETPLACE_URL}\" \
    --output_dir "/novo/projects/departments/mi/lwph/PertRL/results/test_o4_mini/${STAGE}/${DATA_SPLIT}" \
    "
