
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset import DiffExpressionDataset

def create_train_dataloader(csv_dir, tokenizer, batch_size=32, world_size=1, rank=0): # Added tokenizer requirement
    train_dataset = DiffExpressionDataset(csv_dir=csv_dir, split="train")
    return _create_dataloader(train_dataset, tokenizer, batch_size, world_size, rank, is_training=True)

def create_val_dataloader(csv_dir, tokenizer, batch_size=32, world_size=1, rank=0): # Added tokenizer requirement
    val_dataset = DiffExpressionDataset(csv_dir=csv_dir, split="test")
    return _create_dataloader(val_dataset, tokenizer, batch_size, world_size, rank, is_training=False)

def create_test_dataloader(csv_dir, tokenizer, batch_size=32, world_size=1, rank=0): # Added tokenizer requirement
    test_dataset = DiffExpressionDataset(csv_dir=csv_dir, split="test")
    return _create_dataloader(test_dataset, tokenizer, batch_size, world_size, rank, is_training=False)


def _create_dataloader(dataset, tokenizer, batch_size=32, world_size=1, rank=0, is_training=True): # Added tokenizer requirement
    if tokenizer is None:
         raise ValueError("A tokenizer must be provided for _create_dataloader when using GRPO.")

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_training)
    else:
        # Use RandomSampler for training, SequentialSampler for eval/test
        sampler = torch.utils.data.RandomSampler(dataset) if is_training else torch.utils.data.SequentialSampler(dataset)

    def collate_fn(batch):
        # Extract data from batch
        perts = [item["pert"] for item in batch]
        genes = [item["gene"] for item in batch]
        labels = [item["label"] for item in batch] # Numerical labels
        cell_types = [item["cell_type"] for item in batch]
        prompts = [item["prompt"] for item in batch] # List of chat dict lists
        solutions = [item["solution"] for item in batch] # List of "up"/"down"/"no" strings

        # Convert numerical labels to tensor (use long for classification indices)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Apply chat template and tokenize
        # Note: max_length should be appropriate for prompt + expected completion
        # Make sure your tokenizer has a chat template defined!
        # padding='longest' is standard, truncation=True prevents errors on overly long prompts
        try:
            encoded_inputs = tokenizer.apply_chat_template(
                prompts,
                padding="longest",
                truncation=True,
                max_length=1024,  # Adjust as needed (prompt + think + answer)
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True # Crucial for inference models
            )
        except Exception as e:
             # If apply_chat_template fails, maybe manually format and tokenize
             print(f"Warning: tokenizer.apply_chat_template failed: {e}. Falling back to basic tokenization.")
             # Basic fallback (might not respect roles correctly):
             text_prompts = [" ".join([msg['content'] for msg in p]) for p in prompts]
             encoded_inputs = tokenizer(
                 text_prompts, padding="longest", truncation=True, max_length=1024, return_tensors="pt"
             )

        batch_data = {
            "input_ids": encoded_inputs.input_ids,
            "attention_mask": encoded_inputs.attention_mask,
            "labels": labels_tensor,      # Ground truth numerical label (for potential metrics)
            "solution": solutions,        # Ground truth text answer (for reward)
            "perts": perts,
            "genes": genes,
            "cell_types": cell_types,
            "prompts": prompts            # Original prompts (might be useful for debugging)
        }
        return batch_data
    
    pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=4, # Keep using workers for __getitem__
        drop_last=is_training # Drop last incomplete batch during training
    )

    # Note: The original code returned dataloader, sampler.
    # GRPOTrainer usually just needs the dataloader.
    return dataloader 