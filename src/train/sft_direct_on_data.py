from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset
import torch
import wandb
import pandas as pd
from accelerate import Accelerator, PartialState
import os
import logging
import argparse
from typing import Dict, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    max_seq_length: int = 2048
    r: int = 32
    lora_alpha: int = 32
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]



def setup_wandb(args) -> None:
    """Initialize WandB with error handling"""
    if PartialState().is_main_process:
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            run_name = f"gene_expression_{args.task}_lr{args.learning_rate}_bs{args.train_batch_size}_ep{args.num_train_epochs}"
            wandb.init(
                entity=os.getenv('WANDB_ENTITY'),
                project=os.getenv('WANDB_PROJECT'),
                name=run_name,
                config=vars(args)
            )
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            return None

def load_and_prepare_model(args, config: ModelConfig):
    """Load and configure the model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        model = FastModel.get_peft_model(
            model,
            r=config.r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_and_process_data(args, tokenizer, system_prompt: str):
    """Load and process the dataset"""
    logger.info("Processing dataset...")
    try:
        df = pd.read_csv(args.synth_data_csv)
        
        def convert_to_conversational_format(df):
            dataset_rows = []
            for _, row in df.iterrows():
                formated_answer = f"<answer>{row['true_answer']}</answer>"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row['user_prompt']},
                    {"role": "assistant", "content": formated_answer}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                dataset_rows.append({"text": text})
            return Dataset.from_pandas(pd.DataFrame(dataset_rows))

        dataset = convert_to_conversational_format(df)
        dataset = dataset.shuffle(seed=42)
        train_size = int(0.9 * len(dataset))
        
        return {
            'train': dataset.select(range(train_size)),
            'val': dataset.select(range(train_size, len(dataset)))
        }
    except Exception as e:
        logger.error(f"Failed to load or process data: {e}")
        raise



def get_training_args(args) -> TrainingArguments:
    """Configure training arguments"""
    synth_filename = '/'.join(args.synth_data_csv.split('/')[-3])  
    synth_data_name = synth_filename.replace('.csv', '')

    output_dir = f"./output/SFT/{synth_data_name}/lora_COT"
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=5,
        report_to="wandb",
    )

def main(args):
    # Initialize configurations
    model_config = ModelConfig()
    
    # Set up system prompt
    SYSTEM_PROMPT = (
            "You are an molecular and cellular biology expert analyzing gene regulation upon CRISPRi knockdown. "
            "First, provide your reasoning process within <think> </think> tags. Consider relevant pathways "
            "(e.g., cell-type specific biology, ribosome biogenesis, transcription, mitochondrial function, stress response), "
            "gene interactions, and cell-specific context. "
            "Then, choose one option from the following and place your choice within <answer> </answer> tags: 'upregulated', 'downregulated', or 'not differentially expressed'."
            "Example: <think> [Your reasoning here] </think><answer> [upregulated / downregulated / not differentially expressed] </answer>"
        )

    # Setup WandB
    setup_wandb(args)

    try:
        # Load model and tokenizer
        model, tokenizer = load_and_prepare_model(args, model_config)

        # Load and process data
        datasets = load_and_process_data(args, tokenizer, SYSTEM_PROMPT)

        # Print sample examples
        logger.info("=== TRAINING DATA SAMPLES ===")
        for i in range(min(3, len(datasets['train']))):
            logger.info(f"\nExample {i+1}:")
            logger.info(datasets['train'][i]['text'])
            logger.info("-" * 40)

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=get_training_args(args),
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            dataset_text_field="text",
            max_seq_length=model_config.max_seq_length,
            packing=False,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()
        
        synth_filename = args.synth_data_csv.split('/')[-1]  # Get just the filename part
        synth_data_name = synth_filename.replace('.csv', '')

        output_dir = f"./output/SFT/{synth_data_name}/lora_COT"
        # Save model
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()

