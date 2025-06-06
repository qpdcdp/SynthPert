#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --time=24:00:00

# === Define default values using positional parameter expansion ===
#SBATCH --job-name=test_o4_mini

#SBATCH --output=/novo/projects/departments/mi/lwph/PertRL/logs/test_o4/%x_%j.out
#SBATCH --error=/novo/projects/departments/mi/lwph/PertRL/logs/test_o4/%x_%j.err

export TOOL="None" 
export BATCH_SIZE=64
export CELL_LINES="rpe1"
export STAGE="dif"
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


BASE_PATH="/novo/projects/departments/mi/lwph/PertRL"

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
    --test_split_cell_lines "${CELL_LINES}" \
    --batch_size ${BATCH_SIZE} \
    --output_dir "/novo/projects/departments/mi/lwph/PertRL/results/test_o4_mini/${STAGE}/${CELL_LINES}" \
    "