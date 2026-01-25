"""
Evaluation utilities for computing loss metrics.
"""

import torch


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


def calculate_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss across multiple batches from a data loader.
    
    Args:
        data_loader: DataLoader instance
        model: GPT model
        device: Device to run computation on
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Average loss across evaluated batches
    """
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
