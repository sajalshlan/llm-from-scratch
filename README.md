# LLM From Scratch

This repository contains my personal study notes and implementations while learning about Large Language Models. The images used in this project are from Sebastian Raschka's book "Build a Large Language Model (From Scratch)" published by Manning Publications. All image credits go to Sebastian Raschka.

## Structure

### Notebooks
- `self-attention.ipynb` - Self-attention and multi-head attention implementations
- `transformer_block.ipynb` - Transformer block with residual connections
- `gpt-architecture.ipynb` - Complete GPT model architecture
- `training.ipynb` - Training concepts and methodologies
- `evalution.ipynb` - Model evaluation, training loop, and text generation

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
- `generation.py` - Text generation utilities

#### Configuration
- `config.py` - Model configuration (GPT_CONFIG_124M)

## Usage

All components are importable from the `modules` package:

```python
from modules import (
    GPTModel, GPT_CONFIG_124M,
    train_model_simple, evaluate_model,
    calculate_loss_batch, calculate_loss_loader,
    create_dataloader_v1, text_to_token_ids,
    generate_text_simple
)
```

## Data

The training uses "The Verdict" short story, which is automatically downloaded when running the evaluation notebook.
