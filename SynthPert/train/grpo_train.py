from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer # Or your specific tokenizer
import time
import argparse
import wandb
from pathlib import Path
from accelerate import Accelerator, PartialState
import os

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from torch import optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from src.data import DiffExpressionDataset
from src.data import create_train_dataloader, create_test_dataloader, create_val_dataloader
from src.model import build_fsdp_model
from src.train.train_utils import save_checkpoint
from src.rewards.rewards import format_reward_fn, accuracy_reward_fn, simple_reasoning_reward_fn

model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
sft_checkpoint_path = "/workspace/PertRL/output/SFT_full_no_gradient_acc/checkpoint-12864"
use_fsdp=True

mixed_precision = True
if mixed_precision:
    dtype = torch.bfloat16
else:
    dtype = torch.float32

deepspeed_config_dict = {
    "gradient_accumulation_steps": "auto", # Let GRPOConfig/Trainer handle this ideally
    "steps_per_print": "auto", # Let GRPOConfig/Trainer handle logging steps
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        # Consider adding stage3_gather_16bit_weights_on_model_save: true if saving full checkpoints
    },
    "bf16": {
        "enabled": True # MUST match GRPOConfig setting
    },
    "gradient_clipping": "auto", # Let GRPOConfig/Trainer handle this via max_grad_norm
    "train_micro_batch_size_per_gpu": "auto", # MUST be auto or match per_device_train_batch_size
    "wall_clock_breakdown": False
    # It's often recommended to let Trainer handle scheduler settings unless you have specific DeepSpeed scheduler needs
    # "scheduler": {
    #   "type": "WarmupLR",
    #   "params": {
    #      "warmup_min_lr": 0,
    #      "warmup_max_lr": "auto", # Will be set by Trainer
    #      "warmup_num_steps": "auto" # Will be set by Trainer
    #    }
    # },
    # "optimizer": { # Often let Trainer handle this too
    #    "type": "AdamW",
    #    "params": {
    #        "lr": "auto", # Will be set by Trainer
    #        "betas": "auto", # Will be set by Trainer
    #        "eps": "auto", # Will be set by Trainer
    #        "weight_decay": "auto" # Will be set by Trainer
    #    }
    # }
}
# --- Load model TO CPU on ALL ranks ---
# Ensure accelerate or environment doesn't force GPU placement here
model_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": dtype,
    "device_map": None, # Explicitly load without mapping first
    "low_cpu_mem_usage": True, # Try disabling this to force full load
}
model = AutoModelForCausalLM.from_pretrained(
    sft_checkpoint_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    print("Warning: Tokenizer missing pad token; setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
csv_data_directory = "./data" 


embed_layer = model.model.embed_tokens
print(embed_layer.weight.shape)
# 1. Load datasets

train_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="train", exclude_sft_csv = "./output/synth_data/synthetic_data_openai_o3_mini_none_with_critic.csv")
eval_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="test") # Corrected split name likely 'test' or 'validation'


# 2. Configure GRPO Training Arguments
training_args = GRPOConfig(
    output_dir="./grpo_full_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    remove_unused_columns=False,
    logging_steps=1,
    save_strategy="steps",
    reward_weights=[0.5, 1.0],
    save_steps=50,
    report_to=["wandb"],
    push_to_hub=False,
    max_prompt_length=4096,
    max_completion_length=4096,
    num_generations=4,
    beta=0.0,
    temperature=0.9,
    bf16=True,
    use_vllm=False,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_offload_params": False,  # Try without offloading first
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    },
)

state = PartialState()
local_rank = state.local_process_index
if state.is_main_process:
    try:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(entity=os.getenv('WANDB_ENTITY'),project=os.getenv('WANDB_PROJECT'), name="grpo_full_gefion")
    except Exception as e:
        print(f"Error initializing wandb: {e}")
# 3. Instantiate GRPOTrainer
# Combine reward functions (can add weights later if needed)
reward_functions_list = [format_reward_fn, accuracy_reward_fn]

trainer = GRPOTrainer(
    model=model,                 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_functions_list,
)

# 4. Train the model
print("Starting GRPO training...")

train_result = trainer.train()

# 5. Save the final model
trainer.save_model() # Saves model & tokenizer to output_dir
print("Training finished and model saved.")

# Optional: Push to Hub
# if training_args.push_to_hub:
#     trainer.push_to_hub()