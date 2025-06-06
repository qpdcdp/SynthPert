#!/bin/bash

TASK="COT"
SYNTH_DATA_CSV="/path/to/csv/with/synth/cots"


sbatch << EOF
#!/bin/bash
#SBATCH --partition=p4de
#SBATCH --tasks=1
#SBATCH --gres=gpu:1       
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --job-name=unsloth_lora_train_o4_excellent_only_${TASK}
#SBATCH --output=logs/train/unsloth_SFT/%x_%j.log
#SBATCH --error=logs/train/unsloth_SFT/%x_%j.err

# Function to find an empty port
find_empty_port() {
    for port in {1024..65535}; do
        if ! nc -z localhost \$port 2>/dev/null; then
            echo \$port
            return 0
        fi
    done
    echo "no available port found" >&2
    return 1
}
# Set environment variables
export HF_TOKEN="{your HuggingFace token}"
export WANDB_API_KEY="{your wandb api key}"
export WANDB_BASE_URL="{your wandb base url}"
export WANDB_ENTITY="{your wandb entity name}"
export WANDB_PROJECT="{your wandb project name }"

# Get MASTER_NAME and MASTER_ADDR for accelerate launch arguments
export MASTER_NAME="\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR="\$(srun --nodes=1 --ntasks=1 -w "\$MASTER_NAME" hostname --ip-address)"
export MASTER_PORT=\$(find_empty_port)
 
echo "The head node name is \$MASTER_NAME"
echo "The head node IP is \$MASTER_ADDR"
echo "The head node port is \$MASTER_PORT"

# Calculate total GPUs
export SLURM_TOTAL_GPUS=\$((\$SLURM_NNODES * \$SLURM_GPUS_ON_NODE))
echo "Total processes/GPUs: \$SLURM_TOTAL_GPUS"

BASE_PATH="/novo/projects/departments/mi/lwph/PertRL"
export 
# Use srun to properly utilize allocated resources
srun \\
    --export=ALL \\
    bash -c "export PYTHONPATH=\${BASE_PATH}:\$PYTHONPATH && \\
    export PYTHONUNBUFFERED=1 && \\
    source \$(conda info --base)/etc/profile.d/conda.sh && \\
    conda activate unsloth && \\
    cd \${BASE_PATH} && \\
    export PYTHONPATH=\${BASE_PATH} && \\
    python \${BASE_PATH}/main/train.py \\
        --lora \\
        --train_stage SFT \\
        --task ${TASK} \\
        --num_train_epochs 50 \\
        --synth_data_csv ${SYNTH_DATA_CSV}
EOF