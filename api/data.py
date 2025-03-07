import torch
import numpy as np
from tqdm import tqdm

def process_features(sequences):
    """
    Process a list of DNA sequences into a one-hot encoded PyTorch tensor.

    Parameters:
    sequences (list of str): List of DNA sequences, each consisting of 'A', 'T', 'C', 'G'.
                             All sequences must be of the same length.

    Returns:
    torch.Tensor: A tensor of shape (num_sequences, sequence_length, 4) with one-hot encoding.
    """
    # Define the mapping from nucleotides to integers
    mapping = {b'A':0, b'T':1, b'C':2, b'G':3}

    # Get the number of sequences
    num_sequences = len(sequences)

    # Get the sequence length from the first sequence
    sequence_length = 41

    # Check that all sequences have the same length
    if any(len(seq) != sequence_length for seq in sequences):
        raise ValueError("All sequences must be 41 characters long")
    
    byte_array = np.frombuffer(''.join(sequences).encode(), dtype='|S1')
    index_array = np.vectorize(mapping.get, otypes=[np.uint8])(byte_array)
    index_array = index_array.reshape(num_sequences, sequence_length)
    eye_matrix = np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float32)

    return torch.as_tensor(eye_matrix[index_array])