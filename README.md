# LLM From Scratch

This repository contains my personal study notes and implementations while learning about Large Language Models. The images used in this project are from Sebastian Raschka's book "Build a Large Language Model (From Scratch)" published by Manning Publications. All image credits go to Sebastian Raschka.

## Structure

### Notebooks
- `self-attention.ipynb` - Self-attention and multi-head attention implementations
- `transformer_block.ipynb` - Transformer block with residual connections
- `gpt-architecture.ipynb` - Complete GPT model architecture
- `evalution-training.ipynb` - Model evaluation, training loop, and text generation
- `decoding-strategies.ipynb` - Temperature scaling and top-k sampling for text generation
- `loading-gpt2-weights.ipynb` - Loading pretrained GPT-2 weights from OpenAI

### Modules
The `modules/` directory contains reusable Python modules organized by functionality:

#### Core Architecture
- `layers.py` - Basic neural network layers (LayerNorm, GELU, FeedForward)
- `attention.py` - Attention mechanisms (CausalAttention, MultiHeadAttention)
- `transformer.py` - Transformer block implementation
- `gpt.py` - Complete GPT model architecture

#### Training & Evaluation
- `data.py` - Data utilities (tokenization, dataset, dataloaders)
- `evaluation.py` - Loss calculation utilities
- `training.py` - Training loop and model evaluation functions
- `generation.py` - Basic text generation with greedy decoding
- `generation_advanced.py` - Advanced generation with temperature and top-k sampling

#### Pretrained Weights
- `gpt2_download.py` - Download pretrained GPT-2 weights from OpenAI
- `weight_loader.py` - Load pretrained weights into PyTorch model

#### Configuration
- `config.py` - Model configuration (GPT_CONFIG_124M)

## Usage

### Basic Usage

```python
from modules import (
    GPTModel, GPT_CONFIG_124M,
    train_model_simple, evaluate_model,
    create_dataloader_v1, text_to_token_ids,
    generate_text_simple, generate
)

# Basic generation (greedy decoding)
output = generate_text_simple(model, idx, max_new_tokens=50, context_size=1024)

# Advanced generation with temperature and top-k
output = generate(model, idx, max_new_tokens=50, context_size=1024, 
                  temperature=1.5, top_k=50)
```

### Loading Pretrained GPT-2 Weights

```python
from modules.gpt2_download import download_and_load_gpt2
from modules import GPTModel, load_weights_into_gpt

# Download pretrained weights (124M, 355M, 774M, or 1558M)
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# Create model with matching config
config = {
    "vocab_size": settings["n_vocab"],
    "context_length": settings["n_ctx"],
    "emb_dim": settings["n_embd"],
    "num_heads": settings["n_head"],
    "number_of_layers": settings["n_layer"],
    "drop_rate": 0.1,
    "qkv_bias": True
}
model = GPTModel(config)

# Load pretrained weights
load_weights_into_gpt(model, params)
model.eval()
```

## Data

The training uses "The Verdict" short story, which is automatically downloaded when running the evaluation notebook.
