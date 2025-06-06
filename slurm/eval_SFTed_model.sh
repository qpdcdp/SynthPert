#!/bin/bash

# Base directory path
TRAIN_TYPE="dif"
BASE_DIR="/path/to/your/base/dir"
AUROC_MODE="dir" # AUROC_MODE: "dir" = direction of regulation (table 3); "dif" = differentially express or not (table 2)
TYPE="COT"
CKPT_DIR="/path/to/the/unsloth/ckpt/dir/you/want/to/eval"


sbatch << EOF
#!/bin/bash
#SBATCH --partition=p4de
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name=${TYPE}_${CELL_LINE_SPLIT}_AUROC_MODE_${AUROC_MODE}

export MODEL_NAME="DeepSeek-R1-Distill-Llama-8B"
export TOOL="None"
export BATCH_SIZE=24
export TYPE="${TYPE}"
export AUROC_MODE="${AUROC_MODE}"
export CKPT="${CKPT}"
export CKPT_PATH="${CKPT_DIR}"
export CELL_LINES="${CELL_LINE_SPLIT}"
export TRAIN_TYPE="${TRAIN_TYPE}"

BASE_PATH="/path/to/project_dir"

srun \
    --export=ALL \
    bash -c "export PYTHONPATH=\${BASE_PATH}:\\\$PYTHONPATH && \
    export PYTHONUNBUFFERED=1 && \
    source \\\$(conda info --base)/etc/profile.d/conda.sh && \
    conda activate unsloth && \
    cd \${BASE_PATH} && \
    export PYTHONPATH=\${BASE_PATH} && \
    python \${BASE_PATH}/main/test.py \
    --AUROC \
    --AUROC_stage \"\${AUROC_MODE}\" \
    --test_script hf \
    --tool \"\${TOOL}\" \
    --test_split_cell_lines \"\${CELL_LINE_SPLIT}\" \
    --batch_size \${BATCH_SIZE} \
    --model_name \"\${MODEL_NAME}\" \
    --output_dir \"\${BASE_PATH}/results_\${TRAIN_TYPE}/\${CELL_LINE_SPLIT}/\${AUROC_MODE}\" \
    --lora_checkpoint \"\${CKPT_PATH}\"
"

EOF
