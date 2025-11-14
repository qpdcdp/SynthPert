#!/bin/bash
#
#========== Slurm options ==========
#SBATCH --job-name=pert_GRPO
#SBATCH --nodes=3                 # 3 nodes
#SBATCH --ntasks-per-node=1       # one Slurm task per node
#SBATCH --gpus-per-task=8         # 8 GPUs on every node
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16        # tune if you want more CPU threads per GPU
#SBATCH --output=logs/train/GRPO/%x-%j.log
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0-00:00:00
#SBATCH -p cu_0001
#===================================

# ---------- container / paths ----------


# Check if Docker login was successful
if [ $? -ne 0 ]; then
    echo "Docker login failed"
    exit 1
fi

# ---------- CUDA ----------
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_LAUNCH_BLOCKING=0

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

export OMPI_MCA_coll_hcoll_enable=0
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---------- W&B ----------

export WANDB_API_KEY="XXX"
export WANDB_BASE_URL="https://nn-red.wandb.io"
export WANDB_ENTITY="cellular-foundation-model"
export WANDB_PROJECT="Insilico-Perturb-seq"
export WANDB_CACHE_DIR="$SHARED_STORAGE_DATA/wandb/cache"
export WANDB_CONFIG_DIR="$SHARED_STORAGE_DATA/wandb/config"
export WANDB_DIR="$SHARED_STORAGE_DATA/wandb"

# ---------- HuggingFace ----------
# export HF_TOKEN="XXX"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_ETAG_TIMEOUT=500

# ---------- NCCL ----------
export NCCL_DEBUG=DEBUG
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_BUFFSIZE=2097152
export NCCL_NVLS_ENABLE=0

# ---------- veRL / training ----------
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export TRAIN_FILES=/dcai/projects02/users/lwph/PertRL/grpo_parquets/train_GRPO.parquet
export TEST_FILES=/dcai/projects02/users/lwph/PertRL/grpo_parquets/test_GRPO.parquet

export TRAINER_ARGS="\
custom_reward_function.path=/workspace/PertRL/src/rewards/rewards_verl.py \
custom_reward_function.name=overall_reward_fn \
algorithm.adv_estimator=grpo \
data.train_files=${TRAIN_FILES} \
data.val_files=${TEST_FILES} \
data.train_batch_size=768 \
data.max_prompt_length=1024 \
data.max_response_length=2048 \
data.filter_overlong_prompts=True \
data.truncation=error \
actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=192 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
trainer.critic_warmup=0 \
trainer.logger=[console,wandb] \
trainer.project_name=${WANDB_PROJECT} \
trainer.experiment_name=qwen2_14b_inst_function_3_rewards \
trainer.n_gpus_per_node=8 \
trainer.nnodes=${SLURM_NNODES} \
trainer.save_freq=1 \
trainer.test_freq=5 \
trainer.total_epochs=10"

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

# ---------- distributed -- Ray ----------
export MASTER_PORT="$(find_empty_port)"
hostlist=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export MASTER_NAME=${hostlist[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NAME" hostname --ip-address)

# ---------- install deps on ALL nodes ----------
srun -l -w "$SLURM_NODELIST" --kill-on-bad-exit=1 \
  --no-container-mount-home \
  --container-image=$DOCKER_IMAGE \
  --container-name=$SLURM_JOB_NAME \
  --container-mounts=$SHARED_STORAGE:$SHARED_STORAGE,/dcai/projects02/users/lwph/PertRL:/workspace/PertRL \
  --container-workdir=/workspace/PertRL \
  /bin/bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate py_3.11
    pip install -q --no-deps --force-reinstall \
        git+https://github.com/volcengine/verl.git@a43ead6f8253d0af8a06b9df2f0605a8bc6f7621
    pip install -q -r requirements_verl.txt --index-url https://pypi.org/simple/
  "

# ---------- start Ray head ----------
srun --nodes=1 --ntasks=1 -l -w "$MASTER_NAME" \
  --no-container-mount-home \
  --container-image=$DOCKER_IMAGE \
  --container-name=$SLURM_JOB_NAME \
  --container-mounts=$SHARED_STORAGE:$SHARED_STORAGE,/dcai/projects02/users/lwph/PertRL:/workspace/PertRL \
  --container-workdir=/workspace/PertRL \
  /bin/bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate py_3.11
    ray start --head --node-ip-address=$MASTER_ADDR --port=$MASTER_PORT \
        --dashboard-host=0.0.0.0 --dashboard-port=8265 \
        --num-gpus=${SLURM_GPUS_PER_TASK} \
        --node-name ${MASTER_NAME} \
        --block
  " &

sleep 10

# ---------- start Ray workers (every node except the head) ----------
for worker in "${hostlist[@]:1}"; do
  srun --nodes=1 --ntasks=1 -l -w "${worker}" \
    --no-container-mount-home \
    --container-image=$DOCKER_IMAGE \
    --container-name=$SLURM_JOB_NAME \
    --container-mounts=$SHARED_STORAGE:$SHARED_STORAGE,/dcai/projects02/users/lwph/PertRL:/workspace/PertRL \
    --container-workdir=/workspace/PertRL \
    /bin/bash -c "
      source /opt/conda/etc/profile.d/conda.sh
      conda activate py_3.11
      ray start --address=$MASTER_ADDR:$MASTER_PORT \
          --num-gpus=${SLURM_GPUS_PER_TASK} \
          --node-name ${worker} \
          --block
    " &
done

sleep 15

echo "Submitting training job to Ray cluster"
srun --overlap --nodes=1 --ntasks=1 -l -w "$MASTER_NAME" \
  --no-container-mount-home \
  --container-image=$DOCKER_IMAGE \
  --container-name=$SLURM_JOB_NAME \
  --container-workdir=/workspace/PertRL \
  /bin/bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate py_3.11
    ray status --address=$MASTER_ADDR:$MASTER_PORT
    ray job submit --address=http://$MASTER_ADDR:8265 \
        -- python -m verl.trainer.main_ppo \$TRAINER_ARGS \$@
  "

wait   # wait for backgrounded Ray processes so Slurm billing stops correctly
