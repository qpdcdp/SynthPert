import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

# --- 1. CONFIGURE YOUR MODEL DETAILS HERE ---

# The base model your LoRA was trained on. From your config, this is Qwen3-32B.
base_model_id = "Qwen/Qwen3-32B"

# The path to your local LoRA adapter files.
lora_adapter_path = "/novo/projects/departments/mi/lwph/CellPert/output/SFT/COT/default/with_critic/default_split/default_generator_prompt_critic_default_prompt_critic_threshold_excellent_only/lora_COT"

# A local directory to save the merged model before uploading.
merged_model_path = "./merged_model_for_upload"

# ❗ YOUR Hugging Face repo ID (e.g., "your-username/your-model-name").
hf_repo_id = "lhphillips/SFT_single_gene_COT_default_split"


# --- 2. MERGE THE MODEL AND ADAPTER ---

print("Step 1: Merging the base model and LoRA adapter...")

# Load the base model
print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Use "auto" to leverage GPU if available
)

# Load the tokenizer from the adapter, as it may have been modified.
print(f"Loading tokenizer from adapter: {lora_adapter_path}")
tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

# Load the LoRA adapter and merge it into the base model.
print("Loading PEFT adapter and merging...")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged_model = model.merge_and_unload()

# Save the fully merged model to a local directory.
print(f"Saving merged model locally to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("✅ Merging complete.")


# --- 3. UPLOAD THE MERGED MODEL TO HUGGING FACE ---

print("\nStep 2: Uploading the merged model to the Hugging Face Hub...")

# Check if the user has updated the repository ID
if "your-hf-username" in hf_repo_id:
    print("\n⚠️  WARNING: Please update the 'hf_repo_id' variable in the script with your Hugging Face username and desired model name before running!")
else:
    api = HfApi()

    # Create the repository on the Hub.
    print(f"Creating repository '{hf_repo_id}' on the Hub...")
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True, # Don't error if the repo already exists
    )

    # Upload the entire contents of the merged model directory.
    print(f"Uploading files from '{merged_model_path}'...")
    api.upload_folder(
        folder_path=merged_model_path,
        repo_id=hf_repo_id,
        repo_type="model",
    )

    print(f"\n✅ Successfully pushed merged model to: https://huggingface.co/{hf_repo_id}")