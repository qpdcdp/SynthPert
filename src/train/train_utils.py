import os
from pathlib import Path
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullStateDictConfig

def save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, is_best=False, is_final=False):
    """
    Save model checkpoint with FSDP support
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        global_step: Current global step
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        is_final: Whether this is the final checkpoint
    """

    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint name
    if is_final:
        checkpoint_path = os.path.join(checkpoint_dir, "final")
    elif is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "best")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}")
    
    # Create directory for this specific checkpoint
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Get FSDP state dict (full consolidated state)
    if isinstance(model, FSDP):
        # Configure FSDP to gather full state dict
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            model_state = model.state_dict()
            
            # Only rank 0 saves
            if torch.distributed.get_rank() == 0:
                # Save model state in HF format for model.save_pretrained compatibility
                if hasattr(model, "module"):
                    unwrapped_model = model.module
                    if hasattr(unwrapped_model, "save_pretrained"):
                        unwrapped_model.save_pretrained(checkpoint_path)
                    else:
                        torch.save(model_state, os.path.join(checkpoint_path, "pytorch_model.bin"))
                else:
                    torch.save(model_state, os.path.join(checkpoint_path, "pytorch_model.bin"))
                
                # Save optimizer and scheduler states
                opt_state = optimizer.state_dict()
                torch.save(opt_state, os.path.join(checkpoint_path, "optimizer.pt"))
                
                scheduler_state = scheduler.state_dict()
                torch.save(scheduler_state, os.path.join(checkpoint_path, "scheduler.pt"))
                
                # Save training state
                torch.save({
                    "global_step": global_step,
                    "is_best": is_best,
                    "is_final": is_final
                }, os.path.join(checkpoint_path, "training_state.pt"))
                
                print(f"Saved checkpoint to {checkpoint_path}")
    else:
        # For non-FSDP models
        if torch.distributed.get_rank() == 0:
            # Save model state
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(checkpoint_path)
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
            
            # Save optimizer and scheduler states
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
            
            # Save training state
            torch.save({
                "global_step": global_step,
                "is_best": is_best,
                "is_final": is_final
            }, os.path.join(checkpoint_path, "training_state.pt"))
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Synchronize processes to ensure save is complete before proceeding
    torch.distributed.barrier()