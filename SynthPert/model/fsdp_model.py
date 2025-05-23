

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch
)
from functools import partial

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState
import io 

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    TRANSFORMER_LAYER_CLS = LlamaDecoderLayer
    print(f"Using transformer layer class for FSDP wrapping: {TRANSFORMER_LAYER_CLS.__name__}")
except ImportError:
    print("WARNING: Could not import LlamaDecoderLayer. FSDP auto-wrap policy will be None.")
    TRANSFORMER_LAYER_CLS = None




def build_fsdp_model(
    model_name_or_path,
    use_fsdp=True,
    mixed_precision=True,
    bf16=True,
    fsdp_sharding_strategy="FULL_SHARD",
    cpu_offload=False
):
    proc_state = PartialState()
    local_rank = proc_state.local_process_index
    global_rank = proc_state.process_index # For logging
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    if mixed_precision:
        dtype = torch.bfloat16 if bf16 else torch.float16
    else:
        dtype = torch.float32

    # --- Load model TO CPU on ALL ranks ---
    # Ensure accelerate or environment doesn't force GPU placement here
    print(f"[Rank {global_rank}] Loading model '{model_name_or_path}' with device_map=None...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": None, # Explicitly load without mapping first
        "low_cpu_mem_usage": False, # Try disabling this to force full load
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    # --- Force model to CPU after loading (redundant if device_map=None worked, but safe) ---
    model.to("cpu")
    print(f"[Rank {global_rank}] Model loaded. Device check: {next(model.parameters()).device}")


    # --- Convert dtypes (on CPU) ---
    if mixed_precision:
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype)
    model.config.use_cache = False
    if global_rank == 0: # Only print size once
        print_model_size(model, "Model size on CPU (Rank 0)")

    # Barrier to ensure all ranks have loaded the model to CPU
    if dist.is_initialized():
        dist.barrier()
    print(f"[Rank {global_rank}] Passed CPU load barrier.")


    # --- Proceed with FSDP Wrapping ---
    if use_fsdp:
        # ... (Mixed precision policy, CPU offload, sharding strategy setup) ...
        if mixed_precision: mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        else: mixed_precision_policy = None
        cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None
        # ... (Sharding strategy selection) ...
        if fsdp_sharding_strategy == "FULL_SHARD": sharding_strategy = ShardingStrategy.FULL_SHARD
        elif fsdp_sharding_strategy == "SHARD_GRAD_OP": sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif fsdp_sharding_strategy == "NO_SHARD": sharding_strategy = ShardingStrategy.NO_SHARD
        else: sharding_strategy = ShardingStrategy.FULL_SHARD # Default


        # --- Define the Auto Wrap Policy ---
        if TRANSFORMER_LAYER_CLS:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TRANSFORMER_LAYER_CLS})
            print(f"[Rank {global_rank}] Using auto_wrap_policy based on {TRANSFORMER_LAYER_CLS.__name__}")
        else:
            auto_wrap_policy = None
            print(f"[Rank {global_rank}] Using auto_wrap_policy=None (wrapping whole model instance)")

        # --- Wrap with FSDP ---
        print(f"[Rank {global_rank}] Wrapping model with FSDP (Device ID: {local_rank})...")
        try:
            # Pass the CPU model to FSDP
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload_config,
                device_id=local_rank, # FSDP moves params to this CUDA device
                use_orig_params=True,
                # sync_module_states=False, # Default should be False, usually not needed if starting from identical CPU models
            )
        except Exception as e:
            print(f"[Rank {global_rank}] EXCEPTION DURING FSDP WRAPPING: {e}")
            raise e

        # Barrier after wrapping
        if dist.is_initialized():
            dist.barrier()

        if global_rank == 0:
            print("FSDP wrapping complete on all ranks.")
            # (Optional check device after wrapping)
            # print(f"Device check after FSDP (Rank 0): {next(model.parameters()).device}")

        print_fsdp_info(model, proc_state)

    # If not using FSDP, move model to the single designated GPU
    elif not use_fsdp and torch.cuda.is_available():
         model.to(f"cuda:{local_rank}")
         print(f"[Rank {global_rank}] Model moved to cuda:{local_rank} (no FSDP).")


    return model


def print_model_size(model, message="Model"):
    """Prints the total number of parameters in the model with a custom message"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{message}:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


def print_fsdp_info(model, proc_state):
    """Prints information about FSDP sharding"""
    print("\nFSDP Sharding Information:")
    print("=" * 50)
    
    def print_sharding_info(module, prefix=""):
        if isinstance(module, FSDP):
            print(f"{prefix}FSDP Wrapped: {module._get_name()}")
            print(f"{prefix}├── Rank: {proc_state.local_process_index}")
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Count frozen parameters (approximating for quantized params)
            frozen_params = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            
            print(f"{prefix}├── Trainable parameters: {trainable_params:,}")
            print(f"{prefix}├── Frozen parameters: {frozen_params:,}")
            
            # Get GPU memory usage
            gpu_memory = torch.cuda.memory_allocated(device=proc_state.local_process_index)
            gpu_memory_reserved = torch.cuda.memory_reserved(device=proc_state.local_process_index)
            print(f"{prefix}└── GPU Memory: {gpu_memory/1e9:.2f}GB allocated, {gpu_memory_reserved/1e9:.2f}GB reserved")
        
        for name, child in module.named_children():
            print_sharding_info(child, prefix + "    ")
    
    print_sharding_info(model)
    
    # Separate total counts for trainable and frozen parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    if proc_state.is_main_process:
        world_size = proc_state.num_processes
        print(f"\nTotal trainable parameters: {total_trainable:,}")
        print(f"Total frozen parameters: {total_frozen:,}")
        print(f"Expected trainable parameters per rank: ~{total_trainable // world_size:,}")
    print("=" * 50)