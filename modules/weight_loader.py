"""
Utilities for loading pretrained GPT-2 weights into PyTorch model.
"""

import numpy as np
import torch


def assign(left, right):
    """
    Assign weights with shape validation.
    
    Args:
        left: Target parameter tensor
        right: Source weight array
        
    Returns:
        torch.nn.Parameter with the weights
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    Load pretrained GPT-2 weights into PyTorch model.
    
    Args:
        gpt: GPTModel instance
        params: Dictionary of pretrained weights from TensorFlow checkpoint
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformerlayers[b].attention.W_query.weight = assign(
            gpt.transformerlayers[b].attention.W_query.weight, q_w.T)
        gpt.transformerlayers[b].attention.W_key.weight = assign(
            gpt.transformerlayers[b].attention.W_key.weight, k_w.T)
        gpt.transformerlayers[b].attention.W_value.weight = assign(
            gpt.transformerlayers[b].attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformerlayers[b].attention.W_query.bias = assign(
            gpt.transformerlayers[b].attention.W_query.bias, q_b)
        gpt.transformerlayers[b].attention.W_key.bias = assign(
            gpt.transformerlayers[b].attention.W_key.bias, k_b)
        gpt.transformerlayers[b].attention.W_value.bias = assign(
            gpt.transformerlayers[b].attention.W_value.bias, v_b)

        gpt.transformerlayers[b].attention.out_proj.weight = assign(
            gpt.transformerlayers[b].attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformerlayers[b].attention.out_proj.bias = assign(
            gpt.transformerlayers[b].attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformerlayers[b].feedforward.layers[0].weight = assign(
            gpt.transformerlayers[b].feedforward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformerlayers[b].feedforward.layers[0].bias = assign(
            gpt.transformerlayers[b].feedforward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformerlayers[b].feedforward.layers[2].weight = assign(
            gpt.transformerlayers[b].feedforward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformerlayers[b].feedforward.layers[2].bias = assign(
            gpt.transformerlayers[b].feedforward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformerlayers[b].norm1.scale = assign(
            gpt.transformerlayers[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformerlayers[b].norm1.shift = assign(
            gpt.transformerlayers[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformerlayers[b].norm2.scale = assign(
            gpt.transformerlayers[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformerlayers[b].norm2.shift = assign(
            gpt.transformerlayers[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.finalnorm.scale = assign(gpt.finalnorm.scale, params["g"])
    gpt.finalnorm.shift = assign(gpt.finalnorm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
