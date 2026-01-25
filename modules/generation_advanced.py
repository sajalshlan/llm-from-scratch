"""
Advanced text generation utilities with temperature scaling and top-k sampling.
"""

import torch


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using temperature scaling and top-k sampling.
    
    Args:
        model: The GPT model to use for generation
        idx: (batch, n_tokens) array of indices in the current context
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context length the model can handle
        temperature: Temperature for scaling logits (0.0 = greedy, >1.0 = more random)
        top_k: If set, only sample from top k tokens (reduces nonsensical outputs)
        eos_id: End-of-sequence token ID to stop generation early
        
    Returns:
        Generated token indices including the input context
    """
    for _ in range(max_new_tokens):
        # limiting num_tokens in a sequence to our model's context size
        idx_context_capped = idx[:, -context_size :]
        with torch.no_grad():
            logits = model(idx_context_capped)
        logits = logits[:, -1, :] # picking up the possibilites-probabilities for the last token in this sequence so to get the next word

        # top-k sampling
        if top_k is not None:
            # keep only top-k values
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(logits<min_value, torch.tensor(float("-inf")).to(logits.device), logits) # put -inf at places we want 0 prob after softmax to get more sensical outputs when we apply temp scaling

        # temperature scaling
        if temperature > 0.0:
            logits = logits/temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #sample from the distribution
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        
        # append the new token to the existing sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
