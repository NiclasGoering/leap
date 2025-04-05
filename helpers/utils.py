import torch
import torch.nn as nn
import numpy as np
from typing import List, Set, Tuple
import json
from datetime import datetime
import os

# In utils2.py (add this class)
class GPUTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)


def save_dataset(dataset_or_X, y_or_path=None, path=None, rank=0, min_size_bytes=1000):
    """
    Helper function to save dataset with built-in verification.
    
    Can be called in two ways:
    1. save_dataset(dataset_dict, path, rank=0)
       Where dataset_dict contains 'X', 'y' and optionally 'X_test', 'y_test'
    2. save_dataset(X, y, path, rank, min_size_bytes)
       Original call signature
       
    Args:
        dataset_or_X: Either a dictionary containing dataset tensors or X tensor
        y_or_path: Either y tensor or the path (if first arg is a dict)
        path: The file path (used when first arg is X tensor)
        rank: The GPU rank for logging
        min_size_bytes: Minimum file size for verification
    """
    try:
        # Handle both call patterns
        if isinstance(dataset_or_X, dict):
            # Called as save_dataset(dataset_dict, path)
            dataset = dataset_or_X
            save_path = y_or_path
            
            # Ensure dataset has required keys
            if 'X' not in dataset or 'y' not in dataset:
                raise ValueError("Dataset dictionary must contain 'X' and 'y' keys")
                
            # Create save directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Add metadata
            save_dict = {
                'X': dataset['X'].cpu() if isinstance(dataset['X'], torch.Tensor) else dataset['X'],
                'y': dataset['y'].cpu() if isinstance(dataset['y'], torch.Tensor) else dataset['y'],
                'shape_X': dataset['X'].shape,
                'shape_y': dataset['y'].shape,
                'saved_by_rank': rank,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add test data if present
            if 'X_test' in dataset and 'y_test' in dataset:
                save_dict['X_test'] = dataset['X_test'].cpu() if isinstance(dataset['X_test'], torch.Tensor) else dataset['X_test']
                save_dict['y_test'] = dataset['y_test'].cpu() if isinstance(dataset['y_test'], torch.Tensor) else dataset['y_test']
                save_dict['shape_X_test'] = dataset['X_test'].shape
                save_dict['shape_y_test'] = dataset['y_test'].shape
                
            # Save the dataset
            torch.save(save_dict, save_path)
            
            # Verify the save
            if not os.path.exists(save_path):
                raise RuntimeError(f"File does not exist after save: {save_path}")
            if os.path.getsize(save_path) < min_size_bytes:
                raise RuntimeError(f"File too small after save: {save_path} ({os.path.getsize(save_path)} bytes)")
                
            print(f"Rank {rank}: Successfully saved and verified dataset at {save_path}")
            return True
            
        else:
            # Called with original signature: save_dataset(X, y, path, rank, min_size_bytes)
            X = dataset_or_X
            y = y_or_path
            save_path = path
            
            # Create save directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create save dictionary with metadata
            save_dict = {
                'X': X.cpu() if isinstance(X, torch.Tensor) else X,
                'y': y.cpu() if isinstance(y, torch.Tensor) else y,
                'shape_X': X.shape,
                'shape_y': y.shape,
                'saved_by_rank': rank,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save the dataset
            torch.save(save_dict, save_path)
            
            # Verify the save
            if not os.path.exists(save_path):
                raise RuntimeError(f"File does not exist after save: {save_path}")
            if os.path.getsize(save_path) < min_size_bytes:
                raise RuntimeError(f"File too small after save: {save_path} ({os.path.getsize(save_path)} bytes)")
                
            print(f"Rank {rank}: Successfully saved and verified dataset at {save_path}")
            return True
            
    except Exception as e:
        print(f"Rank {rank}: Error saving dataset: {e}")
        return False


def save_results(results: List[dict], results_dir: str, timestamp: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def save_model(model: nn.Module, path: str):
    """Helper function to save model with error handling"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save state dict directly without moving model
        torch.save(model.state_dict(), path)
    except Exception as e:
        print(f"Error saving model: {e}")

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]], device=None):
        self.P = P
        self.sets = sets
        self.device = device
    
    def to(self, device):
        """Move function to specified device"""
        self.device = device
        return self
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        if self.device is not None:
            z = z.to(self.device)
        
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result
    
class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]], device=None):
        self.P = P
        self.sets = sets
        self.device = device
    
    def to(self, device):
        """Move function to specified device"""
        self.device = device
        return self
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        if self.device is not None:
            z = z.to(self.device)
        
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result
    


def generate_master_dataset(P, d, master_size, n_test, msp, seed=42):
    """Generate master training set and test set with fixed seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate master training set
    X_train_master = (2 * torch.bernoulli(0.5 * torch.ones((master_size, d), dtype=torch.float32)) - 1).to(device)
    y_train_master = msp.evaluate(X_train_master)
    
    # Generate test set
    X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), dtype=torch.float32)) - 1).to(device)
    y_test = msp.evaluate(X_test)
    
    return X_train_master, y_train_master, X_test, y_test