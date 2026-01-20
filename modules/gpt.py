"""
GPT Model architecture.
"""

import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .layers import LayerNorm


class GPTModel(nn.Module):
    """GPT model with token and positional embeddings, transformer blocks, and output head."""
    
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.transformerlayers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["number_of_layers"])]
        )
        
        self.finalnorm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_idx):
        batch_size, seq_len = input_idx.shape
        token_embeddings = self.tok_emb(input_idx)
        positional_embeddings = self.pos_emb(torch.arange(seq_len, device=input_idx.device))
        x = token_embeddings + positional_embeddings
        x = self.drop_emb(x)
        x = self.transformerlayers(x)
        x = self.finalnorm(x)
        logits = self.out_head(x)
        return logits
