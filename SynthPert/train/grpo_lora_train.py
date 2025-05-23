from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


import time
import argparse
import wandb
from pathlib import Path
import json
import torch
from accelerate import Accelerator, PartialState
from peft import PeftModel
import os

from src.data import DiffExpressionDataset

from src.train.train_utils import save_checkpoint 
from src.rewards.rewards import  format_reward_fn, accuracy_reward_fn


def main(args):

    # --- Define 4-bit Quantization Config ---
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4", # Recommended quantization type
    #     bnb_4bit_compute_dtype=torch.bfloat16, # Computation dtype during forward/backward pass
    #     bnb_4bit_use_double_quant=True, # Optional: Saves a bit more memory
    # )

    # --- Configuration ---
    deepspeed_config_dict = {
        "gradient_accumulation_steps": "auto", # Let GRPOConfig/Trainer handle this ideally
        "steps_per_print": "auto",
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "offload_optimizer": {
                "device": "none"
            },
            "offload_param": {
                "device": "none"
            },
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
        },
        "bf16": {
            "enabled": True 
        },
        "train_micro_batch_size_per_gpu": "auto", # MUST be auto or match per_device_train_batch_size
        "wall_clock_breakdown": False,
        "gradient_clipping": 1.0,
        "gradient_checkpointing": True, # Enable gradient checkpointing to save memory
        
    }

    accelerator = Accelerator()

    # --- Load Tokenizer ---
    # No changes needed for the tokenizer usually
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important for Causal LM generation

    # --- Load Base Model from  checkpoint ---
    print(f"Rank {accelerator.process_index}: Loading base model (initially on CPU/meta)...")
    if args.lora_checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_checkpoint_path)
        model = model.merge_and_unload()

    if args.sft_checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_checkpoint_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, # Use bfloat16 for model weights
        )
    print("Base model loaded.")

    # --- Define LoRA Config ---
    # Find target modules by printing model architecture: print(model)
    # Common targets for Llama-like models are q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    lora_config = LoraConfig(
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

    # --- Apply LoRA to the Model ---
    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied.")
    model.print_trainable_parameters() 

    # --- Load datasets ---
    csv_data_directory = "./data" # Path to your data
    train_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="train", exclude_sft_csv = args.synth_data_csv) 
    eval_dataset = DiffExpressionDataset(csv_dir=csv_data_directory, train_mode="GRPO", tokenizer= tokenizer, split="test") # Corrected split name likely 'test' or 'validation'


    training_args = GRPOConfig(
        output_dir=args.output_dir, 
        per_device_train_batch_size=18, 
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, # Adjust accumulation to maintain effective batch size (e.g., 4*4*num_gpus = effective batch size)
        optim="adamw_torch", # Use p`a`ged optimizer for memory efficiency with quantization
        learning_rate=1e-4, # LoRA often benefits from a slightly higher learning rate
        lr_scheduler_type="cosine",
        num_train_epochs=1, 
        remove_unused_columns=False, 
        logging_steps=1,
        eval_steps=500, 
        save_strategy="steps",
        save_steps=1000,
        report_to=["wandb"], 
        push_to_hub=False,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended for PEFT/newer PyTorch
        bf16=True, 
        deepspeed=deepspeed_config_dict, 
        # GRPO Specific Parameters
        max_prompt_length=512,
        max_completion_length=2048, 
        num_generations=4, 
        reward_weights=[0.1,1.0],
        temperature=0.9,
    )


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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, 
        reward_funcs=reward_functions_list,
    )

    # --- Train the model ---
    print(f"Starting GRPO training")
    trainer.train()

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
            args.model_name_or_path,
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
        accelerator.wait_for_everyone()

    print("Script finished.")