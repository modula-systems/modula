import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import requests
from typing import Tuple, Dict, Any, Iterator

class TokenDataset:
    """JAX dataset for uint16 tokens."""

    def __init__(self, data_path: str, context_length: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self._length = len(self.data) - self.context_length - 1
    
    def __getitem__(self, idx):
        input_seq = jnp.array(self.data[idx:idx+self.context_length].astype(np.int32))
        target_seq = jnp.array(self.data[idx+1:idx+self.context_length+1].astype(np.int32))
        return input_seq, target_seq

    def __len__(self):
        return self._length

class DataLoader:
    """JAX dataloader for uint16 tokens."""
    
    def __init__(self, dataset: TokenDataset, batch_size: int, shuffle: bool = False, drop_last: bool = True, seed: int = 0):
        """Initialize the dataloader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = jax.random.PRNGKey(seed)
        
    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create an iterator over the dataset."""
        indices = jnp.arange(len(self.dataset))
        
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, indices)
        
        # Calculate number of batches
        if self.drop_last:
            num_batches = len(self.dataset) // self.batch_size
        else:
            num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = indices[start_idx:end_idx]
            
            # Get samples for this batch
            xs, ys = [], []
            for idx in batch_indices:
                x, y = self.dataset[int(idx)]
                xs.append(x)
                ys.append(y)
            
            # Stack into batch
            x_batch = jnp.stack(xs)
            y_batch = jnp.stack(ys)
            
            yield x_batch, y_batch


def download_shakespeare_data(data_dir: str) -> None:
    """Download and prepare the Shakespeare dataset if it doesn't exist.
    Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanogpt
    
    Args:
        data_dir: Directory to store the Shakespeare data
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download the tiny shakespeare dataset
    input_file_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        print("Downloading Shakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    
    # Check if processed files already exist
    if (os.path.exists(os.path.join(data_dir, 'train.bin')) and 
        os.path.exists(os.path.join(data_dir, 'val.bin')) and 
        os.path.exists(os.path.join(data_dir, 'meta.pkl'))):
        return
    
    print("Processing Shakespeare dataset...")
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Create a mapping from characters to integers
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    
    # Create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # Encode both to integers
    train_ids = [stoi[c] for c in train_data]
    val_ids = [stoi[c] for c in val_data]
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    
    # Save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Shakespeare dataset processing complete.")


def load_shakespeare(context_length: int, batch_size: int, shuffle: bool = True) -> Dict[str, Any]:
    """Load the Shakespeare dataset and create dataloaders.
    
    Args:
        context_length: Length of context window for prediction
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the training data
        
    Returns:
        Dictionary containing train_loader, val_loader, and meta information
    """
    # Determine the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'shakespeare')
    
    # Check if the Shakespeare data exists, download if it doesn't
    download_shakespeare_data(data_dir)
    
    # Load meta information
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Create datasets
    train_dataset = TokenDataset(os.path.join(data_dir, 'train.bin'), context_length)
    val_dataset = TokenDataset(os.path.join(data_dir, 'val.bin'), context_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'meta': meta,
        'vocab_size': meta['vocab_size'],
        'encode': lambda s: [meta['stoi'][c] for c in s],
        'decode': lambda l: ''.join([meta['itos'][int(i)] for i in l])
    }


# Example usage
if __name__ == "__main__":
    # Load the data with context length of 8 and batch size of 4
    data = load_shakespeare(context_length=8, batch_size=4)
    
    # Get the first batch from the training loader
    for x_batch, y_batch in data['train_loader']:
        print("Input shape:", x_batch.shape)
        print("Target shape:", y_batch.shape)
        
        # Print the first sequence in the batch
        print("First input sequence:", x_batch[0])
        print("First target sequence:", y_batch[0])
        
        # Decode the first sequence
        print("Decoded input:", data['decode'](x_batch[0]))
        print("Decoded target:", data['decode'](y_batch[0]))
        break 