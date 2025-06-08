conda create -n synthpert python=3.11.11

# Activate the new environment
conda init
conda activate synthpert

# Install the core packages using pip
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3
pip install accelerate==1.5.2
pip install peft==0.15.1
pip install bitsandbytes==0.45.4
pip install numpy==2.2.4
pip install pandas==2.2.3
pip install scikit-learn==1.6.1
pip install tqdm==4.67.1
pip install wandb==0.19.8
pip install datasets==3.5.0
pip install evaluate==0.4.3
pip install unsloth
pip install unsloth-zoo

# Additional core dependencies
pip install safetensors==0.5.3  # Important for model loading
pip install tokenizers==0.21.1   # Required for transformers
pip install xformers==0.0.29.post3  # For memory efficient transformers
pip install einops==0.8.1       # Often used with transformers
pip install deepspeed==0.16.5   # For distributed training
pip install nltk==3.9.1         # For text processing
pip install sentencepiece==0.2.0 # Often needed for tokenization
pip install tiktoken==0.9.0     # For token counting
pip install langchain==0.3.21   # If you're doing LLM applications
pip install openai==1.69.0      # If using OpenAI models

# Development tools
pip install ipykernel          # For Jupyter notebook support
pip install cloudpickle==1.2.2
pip install ruff==0.11.5       # For code linting

pip install fsspec==2024.12.0      # File system operations
pip install pyarrow==19.0.1        # Efficient data handling
pip install aiohttp==3.11.14       # Async HTTP requests
pip install requests==2.32.3        # HTTP requests

# Additional ML/DL tools
pip install diffusers==0.32.2      # If working with diffusion models
pip install scipy==1.15.2          # Scientific computing
pip install sympy==1.13.1          # Symbolic mathematics
pip install filelock==3.18.0       # File locking utilities

# Runtime and optimization
pip install packaging==24.2        # Package metadata handling
pip install typing-extensions==4.14.0 # Type hints for Python 3.11
pip install pydantic==2.11.1       # Data validation
pip install ninja==1.11.1.4        # Build system

# Utilities
pip install psutil==7.0.0          # System monitoring
pip install pillow==11.1.0         # Image processing
pip install protobuf==3.20.3       # Protocol buffers
