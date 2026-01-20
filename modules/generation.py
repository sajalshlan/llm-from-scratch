"""
Text generation utilities for GPT model.
"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding.
    
    Args:
        model: The GPT model to use for generation
        idx: (batch, n_tokens) array of indices in the current context
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context length the model can handle
        
    Returns:
        Generated token indices including the input context
    """
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Limit to our model's context_length
        idx_possible_in_our_context = idx[:, -context_size:]

        # Now inferencing to get the result
        with torch.no_grad():
            logits = model(idx_possible_in_our_context)
        
        # Now we only want to work with the last row as it contains the newly generated word
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Pick only the token id of highest probability term
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx
