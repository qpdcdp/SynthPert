from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import torch
import wandb
import pandas as pd 
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate.utils import DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from src.data import DiffExpressionDataset

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# --- Define 4-bit Quantization Config ---
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4", # Recommended quantization type
#     bnb_4bit_compute_dtype=torch.bfloat16, # Computation dtype during forward/backward pass
#     bnb_4bit_use_double_quant=True, # Optional: Saves a bit more memory
# )

accelerator = Accelerator()

def main(args):

    if args.task == "direct_prediction":
        SYSTEM_PROMPT = (
            "You are an expert bioinformatician. Given a gene perturbation and a cell type, "
            "your task is to list all genes that are upregulated and all genes that are downregulated "
            "as a result of this perturbation.\n"
            "The list of upregulated genes should be prefixed with 'Upregulated: '.\n"
            "The list of downregulated genes should be prefixed with 'Downregulated: '.\n"
            "Each list of genes should be on a new line.\n"
            "Gene names within each list should be enclosed in square brackets and separated by a comma and a space, e.g., ['GENE_A', 'GENE_B'].\n"
            "If no genes fall into a category, use an empty list: [].\n"
            "Wrap your entire response, starting with 'Upregulated:', within <answer> </answer> tags.\n\n"
            "Example of a CORRECT response with genes in both categories:\n"
            "<answer>Upregulated: ['GENE_A', 'GENE_B']\n"
            "Downregulated: ['GENE_C']</answer>\n\n"
            "Example of a CORRECT response with only upregulated genes:\n"
            "<answer>Upregulated: ['GENE_X', 'GENE_Y']\n"
            "Downregulated: []</answer>\n\n"
            "Example of a CORRECT response with no genes in either category:\n"
            "<answer>Upregulated: []\n"
            "Downregulated: []</answer>"
        )
    else:
        SYSTEM_PROMPT = (
            "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
            "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
            "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
            "gene interactions, and cell-specific context. "
            "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed'."
            "Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed] </answer>"
        )

    print("barrier 2 training script")
# --- Load Tokenizer ---
# No changes needed for the tokenizer usually
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important for Causal LM generation

    # --- Load Base Model ---
    print(f"Rank {accelerator.process_index}: Loading base model (initially on CPU/meta)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # quantization_config=quantization_config, # Apply 4-bit config
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
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
    model.print_trainable_parameters() # Verify only LoRA parameters are trainable


    # Load and process CSV
    df = pd.read_csv(args.synth_data_csv)

    # Convert to required format (single text string)
    def convert_to_conversational_format(df):
        """
        Convert a DataFrame with prompt/response columns to chat format for SFTTrainer
        Args: df: pandas DataFrame with 'user_prompt', and 'assistant_response' columns
        Returns: Dataset object with messages in the proper format for apply_chat_template
        """
        dataset_rows = []
        for _, row in df.iterrows():
            messages = []
            # Add system message if present
            print("assistant response: ", row['assistant_response'])
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            # Add user message
            messages.append({"role": "user", "content": row['user_prompt']})
            # Add assistant message
            messages.append({"role": "assistant", "content": row['assistant_response']})
            dataset_rows.append({"messages": messages})
        return Dataset.from_pandas(pd.DataFrame(dataset_rows))

    #eval_dataset = DiffExpressionDataset(csv_dir="./data", split="test") # Corrected split name likely 'test' or 'validation'

    synthetic_dataset = convert_to_conversational_format(df)

    print("=== TRAINING DATA SAMPLES ===")
    for i in range(min(3, len(synthetic_dataset))):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        messages = synthetic_dataset[i]['messages']
        for msg in messages:
            print(f"Role: {msg['role']}")
            print(f"Content: {msg['content']}")
            print("-" * 40)
    # --- Configure Training Arguments ---
    synth_filename = args.synth_data_csv.split('/')[-1]  # Get just the filename part
    synth_data_name = synth_filename.replace('.csv', '')
    
    output_dir = f"./output/SFT/{synth_data_name}/lora"
    print("output_dir", output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        optim="adamw_torch",
        learning_rate=1e-4, # LoRA often benefits from a slightly higher learning rate
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500, # Adjust save frequency as needed
        report_to=["wandb"], # Keep wandb if used
        push_to_hub=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
    )

    model.config.use_cache = False

    # --- Initialize WandB ---
    state = PartialState()
    local_rank = state.local_process_index
    if state.is_main_process:
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(entity=os.getenv('WANDB_ENTITY'),project=os.getenv('WANDB_PROJECT'), name="sft_lora")
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            
    # --- Initialize Trainer ---

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=synthetic_dataset,
        processing_class=tokenizer,
    )

    # --- Train the model ---
    print("Starting training...")
    trainer.train()

    # --- Save the final LoRA adapter ---
    # Saves only the trained LoRA adapter weights, not the full model
    trainer.save_model()
    print(f"Training finished and LoRA adapter saved to {training_args.output_dir}.")

    # To save the merged model (optional, requires more memory/disk):
    print("Merging adapter weights into the base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{training_args.output_dir}/final_merged_model")
    tokenizer.save_pretrained(f"{training_args.output_dir}/final_merged_model")
    print("Merged model saved.")