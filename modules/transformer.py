"""
Transformer block combining attention and feed-forward layers.
"""

import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import LayerNorm, FeedForward


class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and feed-forward layers."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["num_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )
        self.feedforward = FeedForward(config["emb_dim"])
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # First residual block: attention
        skip_connection = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + skip_connection

        # Second residual block: feed-forward
        skip_connection = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + skip_connection

        return x

