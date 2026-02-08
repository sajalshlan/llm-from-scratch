"""
Training utilities for classification fine-tuning.

This module contains functions for training and evaluating 
classification models with PyTorch.
"""

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch.
    
    Args:
        input_batch: Input tensor batch
        target_batch: Target labels batch
        model: The model to evaluate
        device: Device to run on (cpu/cuda)
        
    Returns:
        Loss tensor for the batch
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss across data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        model: The model to evaluate
        device: Device to run on (cpu/cuda)
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Average loss across batches
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Calculate classification accuracy across data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        model: The model to evaluate
        device: Device to run on (cpu/cuda)
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Classification accuracy as a float between 0 and 1
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break
    
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets.
    
    Args:
        model: The model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on (cpu/cuda)
        eval_iter: Number of batches to evaluate
        
    Returns:
        Tuple of (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():  # disables gradient, saves time and memory
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """Train a classification model with periodic evaluation.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        device: Device to run on (cpu/cuda)
        num_epochs: Number of training epochs
        eval_freq: Frequency (in steps) to evaluate model
        eval_iter: Number of batches to use for evaluation
        
    Returns:
        Tuple of (train_losses, val_losses, train_accs, val_accs) lists
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            # optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        
        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs
