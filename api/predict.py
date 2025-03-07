import torch
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress visualization
from typing import Union, List

from .data import process_features  # Custom module for processing sequence features
from .model import Net  # Custom neural network model

from concurrent.futures import ThreadPoolExecutor, as_completed

def load_species_model(species, model_path, device):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = Net(in_channels=4, out_channels=2,
                   use_spectic_conv1d=True,
                   use_spectic_transformer=True).to(device)
        
        model_file = os.path.join(model_path, f'{species}.pt')
        with open(model_file, 'rb') as f:
            if device == "cuda":
                state_dict = torch.load(f, 
                                    map_location=lambda storage, loc: storage.cuda())
            else:
                state_dict = torch.load(f)
        
        model.load_state_dict(state_dict['model_state_dict'])
        if device == "cuda":
            torch.cuda.synchronize()  # 显式同步
        return species, model
    except Exception as e:
        print(f"Error loading {species}: {str(e)}")
        return species, None

class Predictor:
    def __init__(self, model_path=os.path.dirname(os.path.dirname(__file__)), batch_size=128, device=None):
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device == "cuda":
            torch.cuda.init()
        
        self.model_path = os.path.join(model_path, "api/models/")
        # print(self.model_path)
        self.batch_size = batch_size

        # Load species list from model directory, considering only .pt files
        self.species_list = [i[:-3] for i in os.listdir(self.model_path) if i.endswith('.pt')]

        # Initialize and load models for each species
        models = {}
        with ThreadPoolExecutor(max_workers=2) as executor:  # 保守线程数
            futures = {executor.submit(load_species_model, sp, self.model_path, self.device): sp 
                    for sp in self.species_list}
            
            # 带错误处理的进度条
            with tqdm(total=len(futures), desc="Loading models") as pbar:
                for future in as_completed(futures):
                    species, model = future.result()
                    if model is not None:
                        models[species] = model
                    pbar.update(1)
        self.models = models

    def predict(self, species: str, sequences: Union[List[str], str]) -> np.array:
        """
        Predict the class for given sequences using the model for the specified species.

        Parameters:
        -----------
        species (str): The species for which to make predictions.
        sequences (Union[List[str], str]): A list of DNA sequences or a single sequence as a string.

        Returns:
        --------
        np.array: An array of predicted classes for each sequence.

        Raises:
        -------
        ValueError: If the specified species is not found in the models dictionary.
        """
        # Check if the species exists in the models dictionary
        if species not in self.models:
            raise ValueError(f"Unknown species: {species}. Available species: {list(self.models.keys())}")

        # Convert a single string sequence to a list for consistent processing
        if isinstance(sequences, str):
            sequences = [sequences]

        # Process the sequences into features suitable for the model
        features = process_features(sequences)
        # Split features into batches based on the configured batch size
        feature_batches = torch.split(features, self.batch_size)

        # Retrieve the model for the specified species
        model = self.models[species]
        model.eval()  # Set the model to evaluation mode (good practice for inference)

        # List to store model outputs for each batch
        outputs = []
        # Process each batch with a progress bar
        for batch in tqdm(feature_batches, desc="Predicting batches"):
            if self.device == "cuda":
                batch = batch.cuda()  # Move the batch to GPU if avaliable
            with torch.no_grad():  # Disable gradient computation for inference
                output = model(batch)  # Forward pass through the model
                outputs.append(output)  # Store the output

        # Concatenate all batch outputs into a single tensor
        outputs = torch.cat(outputs)
        # Get the predicted class by taking the index of the maximum value along dimension 1
        y_pred = outputs.argmax(dim=1)
        if self.device == "cuda":
            # Detach the tensor from the computation graph, move to CPU, and convert to NumPy array
            return y_pred.detach().cpu().numpy()
        else:
            return y_pred.numpy()