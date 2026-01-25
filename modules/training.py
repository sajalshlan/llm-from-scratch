"""
Training utilities for GPT model.
"""

import torch
from .evaluation import calculate_loss_loader
from .generation import generate_text_simple
from .data import text_to_token_ids, token_ids_to_text


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Simple training loop for GPT model.
    
    Args:
        model: GPT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: Device to run training on
        num_epochs: Number of training epochs
        eval_freq: Frequency of evaluation (in steps)
        eval_iter: Number of batches to use for evaluation
        start_context: Starting text for sample generation
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (train_losses, val_losses, tokens_seen)
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # setting to training mode, because it is also changed to eval mode down which stops dropout

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # calculating loss gradients
            optimizer.step()  # update model weights based on gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # optional evalution step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def calculate_loss_batch(input_batch, output_batch, model, device):
    """
    Calculate cross-entropy loss for a single batch.
    
    Args:
        input_batch: Input token IDs (batch_size, seq_len)
        output_batch: Target token IDs (batch_size, seq_len)
        model: GPT model
        device: Device to run computation on
        
    Returns:
        Loss value for the batch
    """
    input_batch, output_batch = input_batch.to(device), output_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), output_batch.flatten())
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on both training and validation sets.
    
    Args:
        model: GPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run evaluation on
        eval_iter: Number of batches to use for evaluation
        
    Returns:
        Tuple of (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate and print a sample text from the model.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        device: Device to run generation on
        start_context: Starting text for generation
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
