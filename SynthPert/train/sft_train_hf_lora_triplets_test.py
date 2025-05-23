from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM # Ensure DataCollatorForCompletionOnlyLM is imported

import torch
import wandb
import pandas as pd
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate.utils import DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
# from src.data import DiffExpressionDataset # Assuming this is not used for this specific example based on the code

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accelerator = Accelerator()

def main(args):
    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # --- Load Base Model ---
    print(f"Rank {accelerator.process_index}: Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Base model loaded.")

    # --- Define LoRA Config ---
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

    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied.")
    model.print_trainable_parameters()

    # Load and process CSV
    df = pd.read_csv(args.data_path)

    def convert_to_conversational_format(df):
        dataset_rows = []
        for _, row in df.iterrows():
            messages = []
            if pd.notna(row.get('system_prompt')) and row.get('system_prompt'):
                messages.append({"role": "system", "content": row['system_prompt']})
            messages.append({"role": "user", "content": row['user_prompt']})
            messages.append({"role": "assistant", "content": row['assistant_response']})
            dataset_rows.append({"messages": messages})
        return Dataset.from_pandas(pd.DataFrame(dataset_rows))

    synthetic_dataset = convert_to_conversational_format(df)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # + START OF THE VERIFICATION SCRIPT                                          +
    # + You would place the verification code block here.                         +
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    response_template_str = "</think>"
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_str, tokenizer=tokenizer)

    if accelerator.is_main_process: # Optional: Run this check only on the main process
        print("\n--- Starting DataCollatorForCompletionOnlyLM Verification ---")
        
        # Ensure tokenizer.pad_token is set for the collator if it wasn't already


        if len(synthetic_dataset) > 0:
            # Get a sample from your dataset
            sample_index = 0 # Or any other index
            sample = synthetic_dataset[sample_index]
            print(f"\nOriginal sample messages (dataset index {sample_index}):", sample['messages'])

            # Manually apply chat template (SFTTrainer does this internally)
            # Set add_generation_prompt=False as SFTTrainer does for training examples
            # You might need to ensure your tokenizer has a chat_template defined.
            # If not, SFTTrainer might use a default or raise an error.
            # For Llama models, it's usually predefined.
            try:
                formatted_text = tokenizer.apply_chat_template(
                    sample['messages'],
                    tokenize=False,
                    add_generation_prompt=False # SFTTrainer usually sets this to False for training
                )
                print("\nText after apply_chat_template:\n", formatted_text)

                # Tokenize the formatted text
                # Use a max_length that you intend to use for training (e.g., training_args.max_seq_length)
                # or a reasonable value for testing.
                # The collator will handle padding/truncation based on the tokenizer's settings
                # and its own logic if you pass tokenized inputs.
                # However, SFTTrainer typically passes tokenized inputs to the collator.
                
                # 1. Format messages to string
                chat_str = formatted_text # Already done above
                # 2. Tokenize
                tokenized_input = tokenizer(chat_str, truncation=True, max_length=512) # Example max_length

                # The collator expects a list of such tokenized inputs (dictionaries)
                # Each dictionary should have 'input_ids', 'attention_mask', and optionally 'labels'.
                # The collator will create/modify 'labels'.
                # We provide a list containing our single tokenized_input.
                batch = data_collator([tokenized_input]) # Note: tokenized_input is a dict

                print("\nLabels from collator:\n", batch['labels'])
                print("Input IDs from collator:\n", batch['input_ids']) # Check input IDs too
                print("Number of non-ignored labels:", (batch['labels'] != -100).sum().item())

                # Decode the part where labels are not -100
                # Ensure batch['labels'] is a tensor, it should be by default from the collator
                labels_tensor = batch['labels']
                if not isinstance(labels_tensor, torch.Tensor):
                    labels_tensor = torch.tensor(labels_tensor) # Convert if it's a list

                unmasked_labels = labels_tensor[0][labels_tensor[0] != -100]
                if len(unmasked_labels) > 0:
                    print("Decoded unmasked labels (target for loss):\n", tokenizer.decode(unmasked_labels))
                else:
                    print("No unmasked labels found. Check response_template, chat formatting, and tokenization.")
                
                print("\n--- End of DataCollatorForCompletionOnlyLM Verification ---")

            except Exception as e:
                print(f"Error during verification script: {e}")
                print("Make sure your tokenizer has a chat_template defined if using 'messages' format.")
                print("Example tokenizer.chat_template structure for Llama 3:")
                print("""
                {% set loop_messages = messages %}
                {% for message in loop_messages %}
                    {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}
                    {{ content }}
                {% endfor %}
                """)

        else:
            print("Synthetic dataset is empty. Skipping verification script.")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # + END OF THE VERIFICATION SCRIPT                                            +
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # --- Configure Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        # ... (rest of your training_args)
        # Ensure max_seq_length is defined here if used in SFTTrainer
        # max_seq_length=512, # Or whatever value you choose
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        push_to_hub=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        },
    )

    model.config.use_cache = False

    # --- Initialize Trainer ---
    response_template = "<answer>" # Define the response template for SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=synthetic_dataset,
        processing_class=tokenizer, # Explicitly pass the tokenizer
        data_collator=data_collator,
        # SFTTrainer will infer 'messages' column. You can be explicit:
        # dataset_text_field="messages", # if your dataset has a "messages" column
        # Or if you formatted your dataset into a single text string per row, use that column name.                            # This is the max_seq_length SFTTrainer will use for tokenization and collation.
    )

    # --- Train the model ---
    print("Starting training...")
    trainer.train()

    # --- Save the final LoRA adapter ---
    trainer.save_model()
    print(f"Training finished and LoRA adapter saved to {training_args.output_dir}.")

    # To save the merged model (optional):
    if accelerator.is_main_process: # Only main process should do this
        print("Merging adapter weights into the base model...")
        # Ensure model is on CPU for merging if you run into OOM on GPU
        # model = model.cpu() # Uncomment if needed
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(f"{training_args.output_dir}/final_merged_model")
        tokenizer.save_pretrained(f"{training_args.output_dir}/final_merged_model")
        print("Merged model saved.")

# Define a dummy args object for testing if you run this script directly
# In a real scenario, these would come from command-line arguments or a config file
class Args:
    model_name_or_path = "NousResearch/Llama-2-7b-chat-hf" # Replace with a model you have access to
    data_path = "./output/synth_data/synthetic_data_openai_o3_mini_none_with_critic.csv" # Replace with your actual data path
    output_dir = "./output_sft_model"
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 4
    num_train_epochs = 1
    # Add any other args your main function expects

if __name__ == "__main__":
    # Create a dummy CSV for the script to run
    dummy_data = {
        'system_prompt': ["You are a helpful assistant."],
        'user_prompt': ["Explain the water cycle."],
        'assistant_response': ["<think>NEDD4 is an E3 ubiquitin ligase that primarily regulates protein turnover through ubiquitination. There is no evidence that NEDD4 directly controls transcription of PCGF6 in K562 cells or that its substrates include key transcriptional regulators of PCGF6. Since NEDD4 functions post‚Äêtranslationally and PCGF6 expression is driven by Polycomb‚Äêassociated transcriptional networks independent of NEDD4 activity, knocking down NEDD4 is unlikely to change PCGF6 mRNA levels.</think><answer>not differentially expressed</answer>"]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv("your_data.csv", index=False)

    args = Args()
    # You might need to set environment variables for WANDB or comment out wandb parts
    # os.environ["WANDB_API_KEY"] = "your_key"
    # os.environ["WANDB_ENTITY"] = "your_entity"
    # os.environ["WANDB_PROJECT"] = "your_project"

    main(args)