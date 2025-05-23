import os
import time
import argparse
import wandb
from pathlib import Path
from accelerate import PartialState

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch import optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from src.data import DiffExpressionDataset
from src.data import create_train_dataloader, create_test_dataloader, create_val_dataloader
from src.model import build_fsdp_model
from .train_utils import save_checkpoint

def sft_train(
    model,
    train_dataloader,
    train_sampler,
    val_dataloader,
    val_sampler,
    optimizer,
    scheduler,
    device,
    rank,
    world_size,
    epochs=3,
    gradient_accumulation_steps=4,
    max_grad_norm=0.1,
    logging_steps=1,
    eval_steps=100,
    save_steps=100,
    checkpoint_dir="checkpoints",
    bf16=True
):
    """Main training loop"""
    # Create checkpoint directory
    if rank == 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"Training for {epochs} epochs, {len(train_dataloader) // gradient_accumulation_steps * epochs} total steps")
    
    # Initialize tracking variables
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        if rank == 0:
            print(f"Starting epoch {epoch+1}/{epochs}")
        
        # Set epoch for sampler for proper shuffling
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Set model to training mode
        model.train()
        
        # Track metrics for this epoch
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=bf16, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For causal language modeling
                )
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update tracking metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            step_count += 1
            
            # Only update every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if rank == 0 and global_step % logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    
                    metrics = {
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/lr": lr,
                        "train/epoch": epoch + step / len(train_dataloader),
                        "train/step": global_step
                    }
                    
                    print(f"Step {global_step}: loss={metrics['train/loss']:.4f}, lr={lr:.6f}")
                    wandb.log(metrics, step=global_step)
                
                # Evaluation
                if global_step % eval_steps == 0:
                    eval_loss = evaluate(
                        model=model,
                        eval_dataloader=val_dataloader,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        bf16=bf16
                    )
                    
                    # Log evaluation metrics
                    if rank == 0:
                        print(f"Evaluation at step {global_step}: loss={eval_loss:.4f}")
                        wandb.log({"eval/loss": eval_loss}, step=global_step)
                        
                        # Save best model
                        if eval_loss < best_val_loss:
                            best_val_loss = eval_loss
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                checkpoint_dir=checkpoint_dir,
                                is_best=True
                            )
                
                # Save checkpoint
                if rank == 0 and global_step % save_steps == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        checkpoint_dir=checkpoint_dir
                    )
        
        # Calculate average loss for epoch
        epoch_loss = epoch_loss / step_count
        
        # Log epoch metrics
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} completed. Average loss: {epoch_loss:.4f}")
            wandb.log({"train/epoch_loss": epoch_loss, "train/epoch": epoch+1}, step=global_step)
    
    # Save final checkpoint
    if rank == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            checkpoint_dir=checkpoint_dir,
            is_final=True
        )
    
    return global_step, best_val_loss


def evaluate(model, eval_dataloader, device, rank, world_size, bf16=True):
    """Evaluate the model on the validation dataset"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=bf16, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For causal language modeling
                )
                loss = outputs.loss
            
            # Update metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Gather losses across all processes
    if world_size > 1:
        # Create tensors to hold loss and count
        loss_tensor = torch.tensor([total_loss], device=device)
        count_tensor = torch.tensor([total_samples], device=device)
        
        # Gather across all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        total_samples = count_tensor.item()
    
    # Calculate average loss
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    model.train()
    return avg_loss




def main():
    parser = argparse.ArgumentParser(description="Train a model with FSDP")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to model or model name")
    parser.add_argument("--csv_dir", type=str, default="data/cell_types", help="Directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint saving frequency")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--no_fsdp", action="store_true", help="Disable FSDP")
    parser.add_argument("--cpu_offload", action="store_true", help="Use CPU offloading with FSDP")
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"], 
                        help="FSDP sharding strategy")
    parser.add_argument("--run_name", type=str, default=f"fsdp-training-{time.strftime('%Y%m%d-%H%M%S')}", help="Run name")
    args = parser.parse_args()
    
    # Use accelerate's PartialState to get rank information
    proc_state = PartialState()
    rank = proc_state.local_process_index
    world_size = proc_state.num_processes
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize wandb if main process
    if rank == 0:
        try:
            wandb.init(project="gene-expression", name=args.run_name)
            wandb.config.update(args)
        except Exception as e:
            print(f"Error initializing wandb: {e}")
    
    # Build model with FSDP
    model = build_fsdp_model(
        model_name_or_path=args.model_path,
        use_fsdp=not args.no_fsdp,
        mixed_precision=True,
        bf16=args.bf16,
        fsdp_sharding_strategy=args.sharding_strategy,
        cpu_offload=args.cpu_offload
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        model_max_length=6000
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    
    # Create dataloaders
    train_dataloader, train_sampler = create_train_dataloader(
        csv_dir=args.csv_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank
    )
    
    val_dataloader, val_sampler = create_val_dataloader(
        csv_dir=args.csv_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)  # Using transformer-specific beta values
    )
    
    # Create learning rate scheduler
    total_steps = len(train_dataloader) // args.grad_accum * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Log training info
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
        print(f"Model: {args.model_path}")
        print(f"Training examples: {len(train_dataloader.dataset)}")
        print(f"Validation examples: {len(val_dataloader.dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Gradient accumulation steps: {args.grad_accum}")
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print(f"Training for {args.epochs} epochs, {total_steps} steps")
        print(f"Warmup for {warmup_steps} steps")
        print(f"Using precision: {'bf16' if args.bf16 else 'fp32'}")
    
    # Run training loop
    sft_train(
        model=model,
        train_dataloader=train_dataloader,
        train_sampler=train_sampler,
        val_dataloader=val_dataloader,
        val_sampler=val_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=rank,
        world_size=world_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        checkpoint_dir=args.output_dir,
        bf16=args.bf16
    )
    
    # Clean up
    if rank == 0:
        wandb.finish()
    
    # Cleanup distributed environment
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
