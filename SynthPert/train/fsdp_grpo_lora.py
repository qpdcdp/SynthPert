from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import time
import argparse
import wandb
from pathlib import Path
import json
import torch
from accelerate import Accelerator, PartialState
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

import os

from peft import PeftMixedModel

from src.data import DiffExpressionDataset

from src.train.train_utils import save_checkpoint 
from src.rewards.rewards import format_reward_fn, accuracy_reward_fn

# --- Configuration ---
model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
sft_checkpoint_path = "./output/SFT_full_no_gradient_acc/checkpoint-12864" # Path to your SFT checkpoint

accelerator = Accelerator()


# --- build model with FSDP ---

def build_fsdp_model(model_name_or_path, device, use_fsdp=True):
    """
    Loads the LLaMA model with 4-bit quantization and LoRA, then wraps with FSDP.
    """
    # Initialize process state
    proc_state = PartialState()
    local_rank = proc_state.local_process_index
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()


    # Load model with quantization
    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    # Convert any remaining parameters to bfloat16
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    # Disable model caching
    model.config.use_cache = False

    if use_fsdp:

        # Wrap with FSDP

        def _is_peft_model(model):
            classes_to_check = (PeftModel,)
            # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321

            classes_to_check = (*classes_to_check, PeftMixedModel)
            return isinstance(model, classes_to_check)

        # Check if model is PeftModel before FSDP wrapping
        if _is_peft_model(model):
            print("Base model is correctly wrapped with PEFT")
        else:
            print("Warning: Base model is not a PeftModel")

        # Wrap with FSDP first
        wrapped_model = FSDP(
            model,
            # auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
        )
        if _is_peft_model(wrapped_model):
            print("Wrapped model is correctly wrapped with PEFT")
            print(f"Wrapped model class: {wrapped_model.__class__.__name__}")
            print(f"Full class hierarchy: {type(wrapped_model).mro()}")
        else:
            print("Warning: Wrapped model is not a PeftModel")
            print(f"Wrapped model class: {wrapped_model.__class__.__name__}")
            print(f"Full class hierarchy: {type(wrapped_model).mro()}")
        
        print("\nFSDP Sharding Information:")
        print("=" * 50)
        
        def print_sharding_info(module, prefix=""):
            if isinstance(module, FSDP):
                print(f"{prefix}FSDP Wrapped: {module._get_name()}")
                print(f"{prefix}├── Rank: {proc_state.local_process_index}")
                
                # Count trainable (LoRA) parameters (these are in bf16)
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Count frozen parameters (these are in 4-bit)
                frozen_params = sum(p.numel() * 4 if hasattr(p, 'quant_state') else p.numel() 
                                  for p in module.parameters() if not p.requires_grad)
                
                print(f"{prefix}├── Trainable (LoRA) parameters: {trainable_params:,}")
                print(f"{prefix}├── Frozen parameters (unpacked): {frozen_params:,}")
                
                # Get GPU memory usage
                gpu_memory = torch.cuda.memory_allocated(device=proc_state.local_process_index)
                gpu_memory_reserved = torch.cuda.memory_reserved(device=proc_state.local_process_index)
                print(f"{prefix}└── GPU Memory: {gpu_memory/1e9:.2f}GB allocated, {gpu_memory_reserved/1e9:.2f}GB reserved")
            
            for name, child in module.named_children():
                print_sharding_info(child, prefix + "    ")
        
        print_sharding_info(wrapped_model)
        
        # Separate total counts for trainable and frozen parameters
        total_trainable = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in wrapped_model.parameters() if not p.requires_grad)
        
        if proc_state.is_main_process:
            print(f"\nTotal trainable (LoRA) parameters: {total_trainable:,}")
            print(f"Total frozen (base) parameters: {total_frozen:,}")
            print(f"Expected trainable parameters per rank: ~{total_trainable // world_size:,}")
        print("=" * 50)
    
    else:
        print('FSDP SETUP FAILED')
    
    return wrapped_model

def main(rank, world_size, model_path, epochs=1, use_fsdp=True):
    
    device = torch.device(f"cuda:{rank}")
    
    # Build the model (FSDP or DDP)
    model = build_fsdp_model(model_path, device, use_fsdp=use_fsdp)

        # --- Load Tokenizer ---
    # No changes needed for the tokenizer usually
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important for Causal LM generation

    # --- Load datasets ---
    csv_data_directory = "./data" # Path to your data
    train_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="train", exclude_sft_csv = "./output/synth_data/synthetic_data_openai_o3_mini_none_with_critic.csv")
    eval_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="test") # Corrected split name likely 'test' or 'validation'


    training_args = GRPOConfig(
        output_dir=".output/GRPO/lora", # Changed output dir name
        per_device_train_batch_size=12, 
        per_device_eval_batch_size=4,  
        gradient_accumulation_steps=1, # Adjust accumulation to maintain effective batch size (e.g., 4*4*num_gpus = effective batch size)
        optim="paged_adamw_8bit", # Use paged optimizer for memory efficiency with quantization
        learning_rate=1e-4, # LoRA often benefits from a slightly higher learning rate
        lr_scheduler_type="cosine",
        num_train_epochs=1, # Keep epochs as desired
        remove_unused_columns=False, 
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000, # Adjust save frequency as needed
        report_to=["wandb"], # Keep wandb if used
        push_to_hub=False,
        use_vllm=False,
        # GRPO Specific Parameters
        max_prompt_length=2048, # Keep as needed
        max_completion_length=2048, # Reduce if hitting memory limits during generation
        num_generations=4, # Keep as needed
        reward_weights=[0.5, 1.0],
        temperature=0.9, # Keep as needed
        gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended for PEFT/newer PyTorch
        bf16=True, # Enable bf16 training for compute steps (if supported and used in bnb config)
    )

    model.config.use_cache = False

    state = PartialState()
    local_rank = state.local_process_index
    if state.is_main_process:
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(entity=os.getenv('WANDB_ENTITY'),project=os.getenv('WANDB_PROJECT'), name="grpo_lora_gefion")
        except Exception as e:
            print(f"Error initializing wandb: {e}")


    # --- Instantiate GRPOTrainer ---
    # Combine reward functions
    reward_functions_list = [format_reward_fn, accuracy_reward_fn]

    trainer = GRPOTrainer(
        model=model, # Pass the PEFT model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, 
        reward_funcs=reward_functions_list,
    )
    wandb.finish()

    # Cleanup
    dist.destroy_process_group()    
    return trainer

if __name__ == "__main__":

    # --- Train the model ---
    print(f"Starting GRPO training, resuming from SFT checkpoint: {sft_checkpoint_path}")
    train_result = trainer.train()

    # --- Save the final LoRA adapter ---
    # Saves only the trained LoRA adapter weights, not the full model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cpu_model = accelerator.unwrap_model(trainer.model).to('cpu')
        print("Merging adapter weights into the base model on CPU...")
        # merged_model = cpu_model.merge_and_unload() # PeftModel cpu_model already has the adapter merged conceptually
        # Need to actually merge weights
        # Let's reload base model without quantization for clean merge
        print("Reloading base model in full precision for merging...")
        base_model_for_merge = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16, # Or float16/float32
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Loading adapter into full-precision base model...")
        # Load the *final* adapter state from the GRPO output directory
        merged_model = PeftModel.from_pretrained(base_model_for_merge, training_args.output_dir)
        print("Unloading adapter and merging weights...")
        merged_model = merged_model.merge_and_unload()

        save_path = f"{training_args.output_dir}/final_merged_model"
        print(f"Saving merged model to {save_path}...")
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Merged model saved.")
    else:
        # Other processes wait until the main process is done saving.
        accelerator.wait_for_everyone()

    print("Script finished.")