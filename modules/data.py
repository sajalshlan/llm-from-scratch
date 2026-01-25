"""
Data utilities for loading and preprocessing text data.
"""

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs with batch dimension.
    
    Args:
        text: Input text string
        tokenizer: Tokenizer instance (e.g., tiktoken)
        
    Returns:
        Tensor of shape (1, n_tokens) with token IDs
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text.
    
    Args:
        token_ids: Tensor of shape (1, n_tokens) or (n_tokens,)
        tokenizer: Tokenizer instance (e.g., tiktoken)
        
    Returns:
        Decoded text string
    """
    flattened = token_ids.squeeze(0)
    return tokenizer.decode(flattened.tolist())


class GPTDatasetV1(Dataset):
    """
    Dataset for GPT model training using sliding window approach.
    """
    
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialize dataset with sliding window.
        
        Args:
            txt: Raw text data
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    Create a DataLoader for GPT training.
    
    Args:
        txt: Raw text data
        batch_size: Batch size for training
        max_length: Maximum sequence length
        stride: Stride for sliding window
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
