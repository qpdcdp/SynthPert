from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import torch
import wandb
import pandas as pd 
import os
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate.utils import DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from src.data import DiffExpressionDataset
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

accelerator = Accelerator()

# --- Load Tokenizer ---
# No changes needed for the tokenizer usually
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    print("Warning: Tokenizer missing pad token; setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Important for Causal LM generation

# --- Load Base Model ---
print(f"Rank {accelerator.process_index}: Loading base model (initially on CPU/meta)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map=None,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("Base model loaded.")

# --- Define Templates ---
# Make sure these EXACTLY match what the collator expects
# AND how you want your final prompt structured.
# Note: Deepseek chatML format often includes newlines. Check the expected format.
system_prompt_template = "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
user_prompt_template = "<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
assistant_response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_response}<|eot_id|>" # Added newline and EOT

# Load and process CSV
df = pd.read_csv("./output/synth_data/synthetic_data_openai_o3_mini_none_with_critic.csv")

# Convert to required format (single text string)
formatted_data = []
for _, row in df.iterrows():
    text = ""
    if pd.notna(row['system_prompt']) and row['system_prompt']:
         text += system_prompt_template.format(system_prompt=row['system_prompt'])
    text += user_prompt_template.format(user_prompt=row['user_prompt'])
    # IMPORTANT: This is the template the collator looks for
    text += assistant_response_template.format(assistant_response=row['assistant_response'])

    formatted_data.append({"text": text}) # Store in a 'text' field


# Create a dataset
synthetic_dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

#eval_dataset = DiffExpressionDataset(csv_dir="./data", split="test") # Corrected split name likely 'test' or 'validation'



# --- Data Collator ---
instruction_template = "<|start_header_id|>system<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"
    

# Create the completion-only collator
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)


# --- Configure Training Arguments ---

training_args = TrainingArguments(
    output_dir="./output/SFT_full_no_gradient_acc",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_offload_params": True,
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    },
    # Add explicit gradient checkpointing settings
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    save_steps=500, # Adjust save frequency as needed
    logging_steps=1,
    report_to=["wandb"], # Keep wandb if used
    push_to_hub=False,
)
state = PartialState()
local_rank = state.local_process_index
if state.is_main_process:
    try:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(entity=os.getenv('WANDB_ENTITY'),project=os.getenv('WANDB_PROJECT'), name="sft_full_gefion")
    except Exception as e:
        print(f"Error initializing wandb: {e}")
# --- Initialize Trainer ---

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=synthetic_dataset,
    #eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=collator,
)

# --- Train the model ---
print("Starting training...")
trainer.train()

# --- Save the final model weights---
trainer.save_model()
print(f"Training finished and checkpoint saved to {training_args.output_dir}.")
print("Saving tokenizer...")
tokenizer.save_pretrained(f"{training_args.output_dir}/final_merged_model")
