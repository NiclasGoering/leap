#!/usr/bin/env python3
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import sys
import glob
import numpy as np
import json
import yaml
import traceback
from datetime import datetime
from functools import partial
import torch.cuda.amp as amp
from typing import List, Dict, Tuple, Any, Optional
import time
import gzip
import io
import math
import re

# Import your model and helper functions
from helpers.FFNN import DeepNN
from helpers.utils import save_dataset, save_results, save_model

# Ensure prints flush immediately
print = partial(print, flush=True)

# PERFORMANCE CONFIGURATION
# These settings are configurable for different experiment sizes
TINY_THRESHOLD = 1000     # n_train < TINY_THRESHOLD for tiny experiments
SMALL_THRESHOLD = 10000   # TINY_THRESHOLD <= n_train < SMALL_THRESHOLD for small experiments
MEDIUM_THRESHOLD = 100000 # SMALL_THRESHOLD <= n_train < MEDIUM_THRESHOLD for medium experiments
LARGE_THRESHOLD = 2000000 # MEDIUM_THRESHOLD <= n_train < LARGE_THRESHOLD for large experiments
# n_train >= LARGE_THRESHOLD for huge experiments

MAX_PARALLEL_TINY = 16    # For tiny experiments
MAX_PARALLEL_SMALL = 8    # For small experiments
MAX_PARALLEL_MEDIUM = 6   # For medium experiments
MAX_PARALLEL_LARGE = 3    # For large experiments
MAX_PARALLEL_HUGE = 1     # For huge experiments

BATCH_SIZE_TINY = 1024    # Batch size for tiny experiments
BATCH_SIZE_SMALL = 4096   # Batch size for small experiments
BATCH_SIZE_MEDIUM = 8192  # Batch size for medium experiments
BATCH_SIZE_LARGE = 32768  # Batch size for large experiments
BATCH_SIZE_HUGE = 32768   # Batch size for huge experiments (>1M samples)

ORIG_BATCH_SIZE = 32768   # Original batch size reference for LR scaling

BATCH_POWER = 0.5         # Power for batch size scaling

# New constant for evaluation subset size
EVAL_SUBSET_SIZE = 20000  # Maximum number of points to use for evaluation

# Gradient noise scale measurement constants
MEASURE_BATCHES = 25      # Number of batches to measure gradient statistics
MEASURE_BATCH_SIZE = 2056  # Batch size for gradient statistics

# Normalization and standardization functions
def normalize_data(data, dim=1, range_min=-1, range_max=1):
    """Normalize data to range [range_min, range_max]"""
    mins, _ = torch.min(data, dim=dim, keepdim=True)
    maxs, _ = torch.max(data, dim=dim, keepdim=True)
    data_range = maxs - mins
    data_range[data_range == 0] = 1.0
    
    normalized_0_1 = (data - mins) / data_range
    normalized = normalized_0_1 * (range_max - range_min) + range_min
    
    return normalized, {'mins': mins, 'range': data_range, 'range_min': range_min, 'range_max': range_max}

def standardize_data(data, dim=1):
    """Standardize data to mean=0, std=1"""
    means = torch.mean(data, dim=dim, keepdim=True)
    stds = torch.std(data, dim=dim, keepdim=True)
    stds[stds == 0] = 1.0
    
    standardized = (data - means) / stds
    
    return standardized, {'means': means, 'stds': stds}

def measure_gradient_statistics(model, X_train, y_train, device, n_samples=MEASURE_BATCHES, 
                               batch_size=MEASURE_BATCH_SIZE):
    """Measure gradient variance and mean across mini-batches"""
    all_grads = []
    model.eval()
    
    # Generate indices for n_samples mini-batches
    indices_list = []
    for i in range(n_samples):
        if len(X_train) <= batch_size:
            indices = torch.arange(len(X_train), device=device)
        else:
            indices = torch.randperm(len(X_train), device=device)[:batch_size]
        indices_list.append(indices)
    
    # Sample n_samples mini-batches
    for indices in indices_list:
        inputs = X_train[indices]
        targets = y_train[indices]
        
        # Forward and backward pass
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = torch.mean((outputs - targets) ** 2)
        loss.backward()
        
        # Extract and flatten gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1).cpu())
        
        if grads:
            all_grads.append(torch.cat(grads))
    
    if not all_grads:
        return None, None
        
    # Stack gradients from different batches
    batch_grads = torch.stack(all_grads)
    
    # Calculate statistics
    grad_mean = torch.mean(batch_grads, dim=0)
    grad_var = torch.var(batch_grads, dim=0)
    
    return grad_mean, grad_var

def compute_noise_scale(grad_mean, grad_var):
    """Compute gradient noise scale (B*) - the theoretically optimal batch size"""
    if grad_mean is None or grad_var is None:
        return BATCH_SIZE_TINY
        
    trace_V = torch.sum(grad_var)
    squared_mean = torch.sum(grad_mean**2)
    
    # Handle numerical stability
    if squared_mean < 1e-10:
        return BATCH_SIZE_HUGE
        
    B_star = trace_V / squared_mean
    try:
        return max(BATCH_SIZE_TINY, min(BATCH_SIZE_HUGE, int(B_star.item())))
    except (OverflowError, ValueError):
        print(f"Warning: GNS calculation resulted in overflow/infinity, using maximum batch size")
        return BATCH_SIZE_HUGE

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_nn_path(path):
    """Extract parameters from NN dataset path"""
    basename = os.path.basename(path)
    info = {"distribution_type": "gaussian"}  # Mark as gaussian type
    
    # Extract dimension
    d_match = re.search(r'd(\d+)', basename)
    if d_match:
        info['input_dim'] = int(d_match.group(1))
    
    # Extract depth (for k-factor datasets)
    depth_match = re.search(r'depth(\d+)', basename)
    if depth_match:
        info['depth'] = int(depth_match.group(1))
    
    # Extract k-factor 
    k_match = re.search(r'_k([0-9p]+)', basename)
    if k_match:
        k_str = k_match.group(1).replace('p', '.')
        info['k_factor'] = float(k_str)
    
    # Extract architecture
    arch_match = re.search(r'arch([0-9x]+)', basename)
    if arch_match:
        arch_str = arch_match.group(1)
        info['architecture'] = [int(dim) for dim in arch_str.split('x')]
    
    # Extract variance
    var_match = re.search(r'var([0-9p]+)', basename)
    if var_match:
        var_str = var_match.group(1).replace('p', '.')
        info['init_variance'] = float(var_str)
    
    # Extract experiment number
    exp_match = re.search(r'_(\d+)(?:_\d{8}|\.|$)', basename)
    if exp_match:
        info['experiment_num'] = int(exp_match.group(1))
        
    return info

def generate_unique_id(config):
    """Generate a unique identifier for this configuration."""
    ds_name = config['ds_name']
    
    # Format architecture as string
    arch = config.get('architecture', [])
    arch_str = 'x'.join([str(dim) for dim in arch])
    
    # Format initialization variance 
    init_var = config.get('init_variance', 0.1)
    var_str = f"var{init_var}".replace('.', 'p')
    
    # Format k-factor (for k-factor datasets)
    k_factor = config.get('k_factor')
    k_suffix = f"_k{k_factor}".replace('.', 'p') if k_factor is not None else ""
    
    # Format depth (for k-factor datasets)
    ds_depth = config.get('ds_depth')
    depth_suffix = f"_depth{ds_depth}" if ds_depth is not None else ""
    
    align_suffix = "_align" if config['alignment'] else ""
    norm_suffix = "_norm" if config.get('normalize_data', False) else ""
    std_suffix = "_std" if config.get('standardize_data', False) else ""
    
    unique_id = (
        f"{ds_name}"
        f"_d{config['input_dim']}"
        f"{depth_suffix}"
        f"{k_suffix}"
        f"_arch{arch_str}"
        f"_{var_str}"
        f"_h{config['hidden_size']}"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_lr{config['lr']}"
        f"_mode{config['mode']}"
        f"_exp{config['experiment_num']}"
        f"{align_suffix}"
        f"{norm_suffix}"
        f"{std_suffix}"
    )
    
    return unique_id

def find_nn_dataset_files(directory):
    """Find NN dataset X and y files in the given directory"""
    try:
        # Find all .pt.gz files in the directory
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if os.path.isfile(os.path.join(directory, f)) and 
                    f.endswith('.pt.gz')]
        
        print(f"Found {len(all_files)} potential dataset files in {directory}")
        
        # Find NN dataset files (handles both original and k-factor datasets)
        x_files = [f for f in all_files if 'dataset_X_NN_gaussian' in os.path.basename(f)]
        y_files = [f for f in all_files if 'dataset_y_NN_gaussian' in os.path.basename(f)]
        
        if not x_files or not y_files:
            print(f"No NN dataset files found in {directory}")
            return None
        
        # Match X and y files based on the common filename part
        for x_file in x_files:
            x_name = os.path.basename(x_file).replace('dataset_X_', '')
            for y_file in y_files:
                y_name = os.path.basename(y_file).replace('dataset_y_', '')
                if x_name == y_name:
                    print(f"Found matching NN dataset X and y files")
                    return {'x': x_file, 'y': y_file}
        
        # If no matched pairs, use the first of each
        print(f"No exact X/y matches found, using first available files")
        return {'x': x_files[0], 'y': y_files[0]}
        
    except Exception as e:
        print(f"Error finding NN dataset files: {str(e)}")
        return None

def load_dataset_info(directory):
    """Load NN dataset info"""
    if not os.path.isdir(directory):
        return None
    
    # Find dataset files
    dataset_files = find_nn_dataset_files(directory)
    if not dataset_files:
        return None
    
    # Extract parameters
    params = extract_info_from_nn_path(directory)
    
    # Extract dataset name from directory
    ds_name = os.path.basename(directory)
    
    # Calculate size of dataset
    x_size_mb = os.path.getsize(dataset_files['x']) / (1024 * 1024)
    y_size_mb = os.path.getsize(dataset_files['y']) / (1024 * 1024)
    file_size_mb = x_size_mb + y_size_mb
    
    return {
        "files": dataset_files,
        "name": ds_name,
        "params": params,
        "directory": directory,
        "size_mb": file_size_mb
    }

def load_dataset_directly(dataset_files, device):
    """Load NN dataset directly to GPU"""
    data = {'X': None, 'y': None}
    
    try:
        # Load X file
        x_path = dataset_files['x']
        print(f"Loading X data from: {x_path}")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"X file not found: {x_path}")
            
        with gzip.open(x_path, 'rb') as f:
            x_data = torch.load(f, map_location='cpu')
        
        # Check if it's a dictionary with 'X' key or a direct tensor
        if isinstance(x_data, dict) and 'X' in x_data and x_data['X'] is not None:
            data['X'] = x_data['X'].to(device, non_blocking=True)
        elif isinstance(x_data, torch.Tensor):
            data['X'] = x_data.to(device, non_blocking=True)
        else:
            raise ValueError("Could not extract X data from file")
        
        # Load y file
        y_path = dataset_files['y']
        print(f"Loading y data from: {y_path}")
        
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"y file not found: {y_path}")
            
        with gzip.open(y_path, 'rb') as f:
            y_data = torch.load(f, map_location='cpu')
        
        # Check if it's a dictionary with 'y' key or a direct tensor
        if isinstance(y_data, dict) and 'y' in y_data and y_data['y'] is not None:
            data['y'] = y_data['y'].to(device, non_blocking=True)
        elif isinstance(y_data, torch.Tensor):
            data['y'] = y_data.to(device, non_blocking=True)
        else:
            raise ValueError("Could not extract y data from file")
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        raise
    
    # Final checks
    if data['X'] is None:
        raise ValueError("Failed to load X data from files")
    if data['y'] is None:
        raise ValueError("Failed to load y data from files")
    
    # Ensure y has the correct dimensions (squeeze if it's shape [N,1])
    if data['y'] is not None and data['y'].dim() > 1 and data['y'].shape[1] == 1:
        data['y'] = data['y'].squeeze(1)
        print(f"Squeezed y data to shape: {data['y'].shape}")
    
    print(f"Successfully loaded dataset - X shape: {data['X'].shape}, y shape: {data['y'].shape}")
    return data

def generate_all_combinations(config):
    """Generate all parameter combinations."""
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    # Get normalization and standardization flags from base config
    normalize_data = base_cfg.get("normalize_data", False)
    standardize_data = base_cfg.get("standardize_data", False)
    
    all_combinations = []
    
    for sweep_name, sweep_info in sweeps.items():
        # Handle both directory-based and explicit path-based configs
        dataset_paths = sweep_info.get("dataset_paths", [])
        dataset_dirs = sweep_info.get("dataset_dir", [])
        sweep_params = sweep_info.get("parameters", {})
        
        # Handle directory-based approach if specified
        if dataset_dirs:
            expanded_paths = []
            for base_dir in dataset_dirs:
                print(f"Scanning directory: {base_dir}")
                try:
                    # Get all subdirectories that might contain datasets
                    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d))]
                    print(f"Found {len(subdirs)} potential dataset directories")
                    expanded_paths.extend(subdirs)
                except Exception as e:
                    print(f"Error scanning directory {base_dir}: {str(e)}")
            
            # Add discovered paths to dataset_paths
            dataset_paths.extend(expanded_paths)
        
        dataset_infos = []
        for ds_path in dataset_paths:
            ds_info = load_dataset_info(ds_path)
            if ds_info:
                dataset_infos.append(ds_info)
        
        print(f"Loaded {len(dataset_infos)} valid NN datasets for sweep {sweep_name}")
        
        for ds_info in dataset_infos:
            ds_params = ds_info['params']
            input_dim = ds_params.get('input_dim')
            if not input_dim:
                # Try to extract from directory name
                input_dim_match = None
                for part in ds_info['name'].split('_'):
                    if part.startswith('d') and part[1:].isdigit():
                        input_dim = int(part[1:])
                        break
                if not input_dim:
                    print(f"Warning: Could not determine input_dim for {ds_info['directory']}, skipping")
                    continue
            
            # Pass architecture and initialization variance to the combinations
            architecture = ds_params.get('architecture', [])
            init_variance = ds_params.get('init_variance', 0.1)
            
            # Extract k-factor parameters from dataset (new for k-factor datasets)
            depth = ds_params.get('depth')
            k_factor = ds_params.get('k_factor')
            
            # Print dataset info with k-factor if available
            if k_factor is not None:
                print(f"Dataset with k-factor={k_factor}, depth={depth}, architecture={architecture}")
            
            for n_train in sweep_params.get("n_train", [1024]):
                for lr in sweep_params.get("learning_rates", [0.001]):
                    for hidden_size in sweep_params.get("hidden_sizes", [256]):
                        for depth_train in sweep_params.get("depths", [1]):
                            for mode in sweep_params.get("modes", ["standard"]):
                                for alignment in sweep_params.get("alignment", [False]):
                                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                                        combo = {
                                            'dataset_files': ds_info['files'],
                                            'ds_directory': ds_info['directory'],
                                            'ds_name': ds_info['name'],
                                            'hidden_size': hidden_size,
                                            'depth': depth_train,
                                            'input_dim': input_dim,
                                            'n_train': n_train,
                                            'lr': lr,
                                            'mode': mode,
                                            'experiment_num': exp_num,
                                            'base_width': sweep_params.get('base_width', 10),
                                            'alignment': alignment,
                                            'sweep_name': sweep_name,
                                            'architecture': architecture,
                                            'init_variance': init_variance,
                                            'size_mb': ds_info.get('size_mb', 0),
                                            'distribution_type': 'gaussian',
                                            'normalize_data': normalize_data,
                                            'standardize_data': standardize_data,
                                            # New parameters for k-factor datasets
                                            'ds_depth': depth,
                                            'k_factor': k_factor
                                        }
                                        all_combinations.append(combo)
    
    return all_combinations


def worker_process(gpu_id, num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs):
    """Process worker that trains models on a specific GPU"""
    try:
        start_time = time.time()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        print(f"[GPU {gpu_id}] Worker started on device {device}")
        
        # Filter combinations for this GPU using modulo assignment
        worker_combinations = [combo for i, combo in enumerate(all_combinations) if i % num_gpus == gpu_id]
        print(f"[GPU {gpu_id}] Assigned {len(worker_combinations)} configurations")
        
        # Remove already completed configurations
        worker_combinations = [combo for combo in worker_combinations 
                              if generate_unique_id(combo) not in completed_configs]
        
        if not worker_combinations:
            print(f"[GPU {gpu_id}] All configurations already completed")
            return
            
        print(f"[GPU {gpu_id}] Processing {len(worker_combinations)} incomplete configurations")
        
        # Group by dataset directory
        by_dataset = {}
        for combo in worker_combinations:
            ds_dir = combo['ds_directory']
            if ds_dir not in by_dataset:
                by_dataset[ds_dir] = []
            by_dataset[ds_dir].append(combo)
        
        # Process each dataset group
        completed_count = 0
        total_to_process = len(worker_combinations)
        
        for ds_dir, dataset_combos in by_dataset.items():
            # Skip empty dataset groups
            if not dataset_combos:
                continue
                
            print(f"[GPU {gpu_id}] Loading dataset from: {ds_dir}")
            
            try:
                # Load dataset once for all experiments
                dataset_files = dataset_combos[0]['dataset_files']
                data = load_dataset_directly(dataset_files, device)
                
                X_full = data['X']
                y_full = data['y']
                
                print(f"[GPU {gpu_id}] Loaded dataset - X shape: {X_full.shape}, y shape: {y_full.shape}")
                
                # Get normalization and standardization flags from first combo
                should_normalize_data = dataset_combos[0].get('normalize_data', False)
                should_standardize_data = dataset_combos[0].get('standardize_data', False)
                
                # Apply transformations to the FULL dataset before splitting
                transform_params = {}
                
                # Apply normalization if requested (to range [-1, 1])
                if should_normalize_data:
                    print(f"[GPU {gpu_id}] Normalizing data to range [-1, 1]")
                    X_full, norm_params_X = normalize_data(X_full, dim=1, range_min=-1, range_max=1)
                    transform_params['normalize_X'] = norm_params_X
                    
                    # Normalize y data
                    if y_full.dim() > 1 and y_full.shape[1] > 1:
                        y_full, norm_params_y = normalize_data(y_full, dim=1, range_min=-1, range_max=1)
                    else:
                        y_full, norm_params_y = normalize_data(y_full, dim=0, range_min=-1, range_max=1)
                    transform_params['normalize_y'] = norm_params_y
                    
                    print(f"[GPU {gpu_id}] Data normalized. New ranges - X: [{X_full.min().item():.4f}, {X_full.max().item():.4f}], y: [{y_full.min().item():.4f}, {y_full.max().item():.4f}]")
                
                # Apply standardization if requested
                if should_standardize_data:
                    print(f"[GPU {gpu_id}] Standardizing data to mean=0, std=1")
                    X_full, std_params_X = standardize_data(X_full, dim=1)
                    transform_params['standardize_X'] = std_params_X
                    
                    if y_full.dim() > 1 and y_full.shape[1] > 1:
                        y_full, std_params_y = standardize_data(y_full, dim=1)
                    else:
                        y_full, std_params_y = standardize_data(y_full, dim=0)
                    transform_params['standardize_y'] = std_params_y
                    
                    print(f"[GPU {gpu_id}] Data standardized. New stats - X: mean={X_full.mean().item():.4f}, std={X_full.std().item():.4f}, y: mean={y_full.mean().item():.4f}, std={y_full.std().item():.4f}")
                
                # Group experiments by size for optimal batching
                tiny_exps = [c for c in dataset_combos if c['n_train'] < TINY_THRESHOLD]
                small_exps = [c for c in dataset_combos if TINY_THRESHOLD <= c['n_train'] < SMALL_THRESHOLD]
                medium_exps = [c for c in dataset_combos if SMALL_THRESHOLD <= c['n_train'] < MEDIUM_THRESHOLD]
                large_exps = [c for c in dataset_combos if MEDIUM_THRESHOLD <= c['n_train'] < LARGE_THRESHOLD]
                huge_exps = [c for c in dataset_combos if c['n_train'] >= LARGE_THRESHOLD]
                
                # Process tiny experiments in parallel batches
                if tiny_exps:
                    for i in range(0, len(tiny_exps), MAX_PARALLEL_TINY):
                        batch = tiny_exps[i:i+MAX_PARALLEL_TINY]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(tiny_exps)} tiny experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_TINY, 10, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                            transform_params
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # Process small experiments in parallel batches
                if small_exps:
                    for i in range(0, len(small_exps), MAX_PARALLEL_SMALL):
                        batch = small_exps[i:i+MAX_PARALLEL_SMALL]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(small_exps)} small experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_SMALL, 20, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                            transform_params
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # Process medium experiments in smaller parallel batches
                if medium_exps:
                    for i in range(0, len(medium_exps), MAX_PARALLEL_MEDIUM):
                        batch = medium_exps[i:i+MAX_PARALLEL_MEDIUM]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(medium_exps)} medium experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_MEDIUM, 30, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                            transform_params
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # Process large experiments with less parallelism
                if large_exps:
                    for i in range(0, len(large_exps), MAX_PARALLEL_LARGE):
                        batch = large_exps[i:i+MAX_PARALLEL_LARGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(large_exps)} large experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_LARGE, 40, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                            transform_params
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # Process huge experiments individually
                if huge_exps:
                    for i in range(0, len(huge_exps), MAX_PARALLEL_HUGE):
                        batch = huge_exps[i:i+MAX_PARALLEL_HUGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(huge_exps)} huge experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_HUGE, 50, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                            transform_params
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR processing dataset {ds_dir}: {str(e)}")
                traceback.print_exc()
                continue
            
            # Clear memory after processing a dataset
            del X_full, y_full, data
            torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f"[GPU {gpu_id}] Completed all experiments in {elapsed_time:.2f} seconds")
        print(f"[GPU {gpu_id}] Average time per experiment: {elapsed_time/max(1, completed_count):.2f} seconds")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()


def convert_tensors_for_json(obj):
    """Convert PyTorch tensors to Python native types for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_for_json(item) for item in obj]
    else:
        return obj


def fast_parallel_training(config_batch, device, X_full, y_full, base_config, 
                          initial_batch_size, eval_interval, max_epochs,
                          full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER,
                          transform_params=None):
    """Train multiple models in parallel with Gradient Noise Scale optimization"""
    # Define GNS measurement schedule - early and logarithmically spaced points
    GNS_EPOCHS = [5]  # Always measure at epoch 5
    
    if max_epochs > 10:
        for i in range(1, 10):
            epoch = min(max_epochs - 1, int(5 + (max_epochs - 5) * (i / 9)**2))
            if epoch not in GNS_EPOCHS:
                GNS_EPOCHS.append(epoch)
        GNS_EPOCHS.sort()
    
    LATER_ADJUSTMENTS_STRENGTH = 0.8  # Factor for later adjustments
    
    print(f"[GPU {gpu_id}] GNS will be measured at epochs: {GNS_EPOCHS}")
    
    # Get test data
    n_test = base_config['n_test']
    fixed_seed = abs(hash(str(config_batch[0]['ds_directory']))) % (2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(fixed_seed)
    indices = torch.randperm(len(X_full), device=device, generator=generator)
    test_indices = indices[:n_test]
    train_master_indices = indices[n_test:]
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]
    
    # Setup for parallel training
    models = []
    optimizers = []
    schedulers = []
    original_lrs = []
    train_data = []
    eval_data = []
    batch_sizes = []
    config_items = []
    unique_ids = []
    early_stop_flags = []
    
    # Initialize all models
    for config_item in config_batch:
        unique_id = generate_unique_id(config_item)
        
        if unique_id in completed_configs:
            continue
            
        # Sample training data
        n_train = config_item['n_train']
        sample_seed = hash(f"sample_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        torch.manual_seed(sample_seed)
        
        if n_train < len(train_master_indices):
            train_indices = train_master_indices[torch.randperm(len(train_master_indices), device=device)[:n_train]]
            X_train = X_full[train_indices]
            y_train = y_full[train_indices]
        else:
            X_train = X_full[train_master_indices]
            y_train = y_full[train_master_indices]
        
        # Create evaluation subset (limited to EVAL_SUBSET_SIZE points)
        eval_seed = hash(f"eval_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        generator = torch.Generator(device=device)
        generator.manual_seed(eval_seed)
        
        if len(X_train) <= EVAL_SUBSET_SIZE:
            X_eval = X_train
            y_eval = y_train
        else:
            eval_indices = torch.randperm(len(X_train), device=device, generator=generator)[:EVAL_SUBSET_SIZE]
            X_eval = X_train[eval_indices]
            y_eval = y_train[eval_indices]
        
        # Save dataset if requested
        if base_config.get('save_dataset', False):
            dataset_dir = os.path.join(full_results_dir, "datasets")
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.join(dataset_dir, f"dataset_{unique_id}.pt")
            dataset = {
                'X': X_train.cpu(), 
                'y': y_train.cpu(), 
                'X_test': X_test.cpu(), 
                'y_test': y_test.cpu(),
                'transform_params': transform_params
            }
            save_dataset(dataset, dataset_path)
        
        # Initialize model
        model_seed = hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}")
        torch.manual_seed(model_seed)
        
        input_dim = config_item['input_dim']
        hidden_size = config_item['hidden_size']
        depth = config_item['depth']
        
        model = DeepNN(
            input_dim, 
            hidden_size, 
            depth, 
            mode=config_item['mode'], 
            alignment=config_item['alignment'],
            base_width=config_item.get('base_width', 10)
        ).to(device)
        
        # Use the learning rate from config file as base learning rate
        base_lr = config_item["lr"]
        
        # Start with initial batch size based on dataset size
        if n_train < TINY_THRESHOLD:
            current_batch_size = BATCH_SIZE_TINY
        elif n_train < SMALL_THRESHOLD:
            current_batch_size = BATCH_SIZE_SMALL
        elif n_train < MEDIUM_THRESHOLD:
            current_batch_size = BATCH_SIZE_MEDIUM
        elif n_train < LARGE_THRESHOLD:
            current_batch_size = BATCH_SIZE_LARGE
        else:
            current_batch_size = BATCH_SIZE_HUGE
        
        # Scale with batch size
        batch_size_ratio = current_batch_size / ORIG_BATCH_SIZE
        scaled_lr = base_lr * (batch_size_ratio ** BATCH_POWER)
        
        # Store the original learning rate for warmup
        original_lr = scaled_lr
        
        # Start with a very small learning rate for extended warmup
        initial_warmup_lr = scaled_lr * 0.01  # Start at 1% of target LR
        
        # Create optimizer with weight decay for regularization
        weight_decay = float(base_config["weight_decay"])
        optimizer = optim.Adam(model.parameters(), lr=initial_warmup_lr, weight_decay=weight_decay, eps=1e-9)
        
        # Use CosineAnnealingLR for smooth learning rate decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=scaled_lr * 0.3
        )
        
        # Store everything
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        original_lrs.append(original_lr)
        train_data.append((X_train, y_train))
        eval_data.append((X_eval, y_eval))
        batch_sizes.append(current_batch_size)
        config_items.append(config_item)
        unique_ids.append(unique_id)
        early_stop_flags.append(False)
    
    if not models:  # All experiments were already completed
        return 0
    
    # Parallel training with BF16 mixed precision
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Error history tracking
    train_errors = [[] for _ in range(len(models))]
    test_errors = [[] for _ in range(len(models))]
    epoch_numbers = [[] for _ in range(len(models))]
    
    # Implement a longer warmup period for high-degree polynomials
    warmup_epochs = max(5, int(max_epochs * 0.3))
    print(f"[GPU {gpu_id}] Using extended warmup period of {warmup_epochs} epochs ({warmup_epochs/max_epochs:.1%} of training)")
    
    # Track initial errors
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            for i, model in enumerate(models):
                if early_stop_flags[i]:
                    continue
                    
                X_eval, y_eval = eval_data[i]
                model.eval()
                
                eval_output = model(X_eval)
                test_output = model(X_test)
                
                train_error = torch.mean((eval_output - y_eval) ** 2).item()
                test_error = torch.mean((test_output - y_test) ** 2).item()
                
                train_errors[i].append(train_error)
                test_errors[i].append(test_error)
                epoch_numbers[i].append(0)
                
                # Save initial model if requested
                if base_config.get('save_model', False):
                    initial_model_dir = os.path.join(full_results_dir, "initial_models")
                    os.makedirs(initial_model_dir, exist_ok=True)
                    initial_model_path = os.path.join(initial_model_dir, f"initial_model_{unique_ids[i]}.pt")
                    save_model(model, initial_model_path)
    
    # Track GNS measurements 
    gns_measured_epochs = [[] for _ in range(len(models))]
    
    # Early stopping parameters
    early_stop_threshold = 1e-12
    early_stop_patience = 200
    best_errors = [float('inf') for _ in range(len(models))]
    patience_counters = [0 for _ in range(len(models))]
    
    # Fast parallel training loop
    for epoch in range(max_epochs):
        # Check if all models have early stopped
        if all(early_stop_flags):
            break
        
        # Apply gradual warmup if in warmup phase
        if epoch < warmup_epochs:
            warmup_progress = epoch / warmup_epochs
            for i, optimizer in enumerate(optimizers):
                if early_stop_flags[i]:
                    continue
                warmup_factor = warmup_progress ** 3  # Cubic curve gives slower initial warmup
                target_lr = original_lrs[i]
                current_lr = target_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                log_interval = min(int(warmup_epochs / 10), 100) 
                if epoch % log_interval == 0 or epoch == warmup_epochs - 1:
                    print(f"[GPU {gpu_id}] Model {i}: Warmup progress {warmup_progress:.2f}, LR = {current_lr:.8f} (target: {target_lr:.8f})")
        
        # Measure gradient noise scale at the specified epochs
        if epoch in GNS_EPOCHS:
            for i, model in enumerate(models):
                if early_stop_flags[i]:
                    continue
                    
                X_train, y_train = train_data[i]
                
                # Measure gradient statistics
                print(f"[GPU {gpu_id}] Measuring GNS for model {i} at epoch {epoch}/{max_epochs}")
                grad_mean, grad_var = measure_gradient_statistics(model, X_train, y_train, device)
                
                if grad_mean is not None and grad_var is not None:
                    # Compute optimal batch size
                    try:
                        optimal_batch_size = compute_noise_scale(grad_mean, grad_var)
                        if math.isinf(optimal_batch_size):
                            print(f"[GPU {gpu_id}] Model {i}: GNS returned infinity, using max batch size: {BATCH_SIZE_HUGE}")
                            optimal_batch_size = BATCH_SIZE_HUGE
                    except (OverflowError, ValueError):
                        print(f"[GPU {gpu_id}] Model {i}: GNS calculation resulted in overflow, using max batch size: {BATCH_SIZE_HUGE}")
                        optimal_batch_size = BATCH_SIZE_HUGE
                        
                    print(f"[GPU {gpu_id}] Model {i}: Measured GNS gives optimal batch size: {optimal_batch_size}")
                    
                    # Apply dampening for later measurements
                    if len(gns_measured_epochs[i]) > 0:
                        adjustment_strength = LATER_ADJUSTMENTS_STRENGTH
                        old_batch_size = batch_sizes[i]
                        blended_size = int(old_batch_size * (1 - adjustment_strength) + 
                                          optimal_batch_size * adjustment_strength)
                        optimal_batch_size = blended_size
                        print(f"[GPU {gpu_id}] Model {i}: Blended batch size: {optimal_batch_size} (old: {old_batch_size})")
                    
                    # Adjust batch size (capped for memory constraints)
                    if X_train.shape[0] < TINY_THRESHOLD:
                        optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_TINY)
                    elif X_train.shape[0] < SMALL_THRESHOLD:
                        optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_SMALL)
                    elif X_train.shape[0] < MEDIUM_THRESHOLD:
                        optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_MEDIUM)
                    elif X_train.shape[0] < LARGE_THRESHOLD:
                        optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_LARGE)
                    else:
                        optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_HUGE)
                    
                    # Ensure batch size is reasonable relative to dataset size
                    optimal_batch_size = min(optimal_batch_size, len(X_train))
                    
                    # Update batch size
                    old_batch_size = batch_sizes[i]
                    batch_sizes[i] = optimal_batch_size
                    
                    # Check if batch size changed significantly (more than 10%)
                    if abs(old_batch_size - optimal_batch_size) / old_batch_size > 0.1:
                        # Adjust learning rate based on new batch size
                        batch_size_ratio = optimal_batch_size / old_batch_size
                        old_lr = optimizers[i].param_groups[0]['lr']
                        new_lr = old_lr * (batch_size_ratio ** BATCH_POWER)
                        
                        # If in warmup, adjust the stored original learning rate too
                        if epoch < warmup_epochs:
                            original_lrs[i] = original_lrs[i] * (batch_size_ratio ** BATCH_POWER)
                            # Recompute current warmup LR
                            warmup_progress = epoch / warmup_epochs
                            warmup_factor = warmup_progress ** 3
                            new_lr = original_lrs[i] * warmup_factor
                        
                        # Update optimizer learning rate
                        for param_group in optimizers[i].param_groups:
                            param_group['lr'] = new_lr
                        
                        print(f"[GPU {gpu_id}] Model {i}: Updated batch size from {old_batch_size} to {optimal_batch_size}")
                        print(f"[GPU {gpu_id}] Model {i}: Updated LR from {old_lr:.6f} to {new_lr:.6f}")
                    else:
                        print(f"[GPU {gpu_id}] Model {i}: Batch size change too small, keeping at {batch_sizes[i]}")
                    
                    # Track the epoch where GNS was measured
                    gns_measured_epochs[i].append(epoch)
        
        # Train each model with one batch
        for i, model in enumerate(models):
            if early_stop_flags[i]:
                continue
                
            model.train()
            optimizer = optimizers[i]
            X_train, y_train = train_data[i]
            batch_size = batch_sizes[i]
            
            # Sample random batch
            if len(X_train) <= batch_size:
                batch_X, batch_y = X_train, y_train
            else:
                batch_indices = torch.randperm(len(X_train), device=device)[:batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
            
            # One training step
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Step the scheduler if after warmup
            if epoch >= warmup_epochs:
                # For first epoch after warmup, reset the scheduler with proper LR
                if epoch == warmup_epochs:
                    current_lr = optimizer.param_groups[0]['lr']
                    schedulers[i] = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max_epochs - warmup_epochs,
                        eta_min=current_lr * 0.3
                    )
                    print(f"[GPU {gpu_id}] Model {i}: Warmup complete, starting scheduler at LR={current_lr:.8f}")
                
                # Step the scheduler
                schedulers[i].step()
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    for i, model in enumerate(models):
                        if early_stop_flags[i]:
                            continue
                            
                        X_eval, y_eval = eval_data[i]
                        model.eval()
                        
                        eval_output = model(X_eval)
                        test_output = model(X_test)
                        
                        train_error = torch.mean((eval_output - y_eval) ** 2).item()
                        test_error = torch.mean((test_output - y_test) ** 2).item()
                        
                        train_errors[i].append(train_error)
                        test_errors[i].append(test_error)
                        epoch_numbers[i].append(epoch + 1)
                        
                        # Early stopping logic
                        if train_error < best_errors[i]:
                            best_errors[i] = train_error
                            patience_counters[i] = 0
                        else:
                            patience_counters[i] += 1
                        
                        if train_error < early_stop_threshold or patience_counters[i] >= early_stop_patience:
                            early_stop_flags[i] = True
    
    # Add fine tuning phase
    fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
    
    # Define GNS measurement schedule for fine-tuning
    FT_GNS_EPOCHS = [5]
    
    if fine_tuning_epochs > 10:
        for i in range(1, 5):
            epoch = min(fine_tuning_epochs - 1, int(5 + (fine_tuning_epochs - 5) * (i / 4)**2))
            if epoch not in FT_GNS_EPOCHS:
                FT_GNS_EPOCHS.append(epoch)
        FT_GNS_EPOCHS.sort()
    
    print(f"[GPU {gpu_id}] Fine-tuning GNS will be measured at epochs: {FT_GNS_EPOCHS}")
    
    # Add warmup for fine-tuning phase
    ft_warmup_epochs = max(5, int(fine_tuning_epochs * 0.2))
    print(f"[GPU {gpu_id}] Using fine-tuning warmup of {ft_warmup_epochs} epochs ({ft_warmup_epochs/fine_tuning_epochs:.1%} of fine-tuning)")
    
    # Do fine-tuning phase for all models
    for i, model in enumerate(models):
        if early_stop_flags[i] and best_errors[i] > early_stop_threshold:  # Only do fine-tuning if not already converged
            X_train, y_train = train_data[i]
            X_eval, y_eval = eval_data[i]
            optimizer = optimizers[i]
            batch_size = batch_sizes[i]
            
            # Switch to full precision for perfect memorization
            model_dtype = next(model.parameters()).dtype
            
            # Convert model and data to float32
            model.to(dtype=torch.float32)
            X_train_fp32 = X_train.to(dtype=torch.float32)
            y_train_fp32 = y_train.to(dtype=torch.float32)
            X_eval_fp32 = X_eval.to(dtype=torch.float32)
            y_eval_fp32 = y_eval.to(dtype=torch.float32)
            
            # Set up fine-tuning with warm-up
            current_lr = optimizer.param_groups[0]['lr']
            initial_ft_lr = current_lr * 0.01
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_ft_lr
            
            # Create scheduler for after warmup
            ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=fine_tuning_epochs - ft_warmup_epochs,
                eta_min=current_lr / 100
            )
            
            # Track fine-tuning GNS measurements
            ft_gns_measured_epochs = []
            
            # Fine-tuning loop
            for ft_epoch in range(fine_tuning_epochs):
                # Apply warmup if in warmup phase
                if ft_epoch < ft_warmup_epochs:
                    warmup_progress = ft_epoch / ft_warmup_epochs
                    warmup_factor = warmup_progress ** 3
                    new_lr = current_lr * warmup_factor
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    if ft_epoch % 10 == 0 or ft_epoch == ft_warmup_epochs - 1:
                        print(f"[GPU {gpu_id}] Model {i}: Fine-tuning warmup {warmup_progress:.2f}, LR = {new_lr:.8f}")
                
                # Measure gradient noise scale at specified epochs
                if ft_epoch in FT_GNS_EPOCHS:
                    print(f"[GPU {gpu_id}] Measuring fine-tuning GNS for model {i} at epoch {ft_epoch}/{fine_tuning_epochs}")
                    grad_mean, grad_var = measure_gradient_statistics(model, X_train_fp32, y_train_fp32, device)
                    
                    if grad_mean is not None and grad_var is not None:
                        try:
                            optimal_batch_size = compute_noise_scale(grad_mean, grad_var)
                            if math.isinf(optimal_batch_size):
                                print(f"[GPU {gpu_id}] Model {i}: Fine-tuning GNS returned infinity, using max batch size: {BATCH_SIZE_HUGE}")
                                optimal_batch_size = BATCH_SIZE_HUGE
                        except (OverflowError, ValueError):
                            print(f"[GPU {gpu_id}] Model {i}: Fine-tuning GNS calculation resulted in overflow, using max batch size: {BATCH_SIZE_HUGE}")
                            optimal_batch_size = BATCH_SIZE_HUGE
                            
                        print(f"[GPU {gpu_id}] Model {i}: Fine-tuning GNS gives optimal batch size: {optimal_batch_size}")
                        
                        # Apply dampening for later measurements
                        if len(ft_gns_measured_epochs) > 0:
                            adjustment_strength = LATER_ADJUSTMENTS_STRENGTH
                            old_batch_size = batch_size
                            blended_size = int(old_batch_size * (1 - adjustment_strength) + 
                                              optimal_batch_size * adjustment_strength)
                            optimal_batch_size = blended_size
                            print(f"[GPU {gpu_id}] Model {i}: Blended fine-tuning batch size: {optimal_batch_size} (old: {old_batch_size})")
                        
                        # Adjust batch size (capped for memory constraints)
                        if X_train_fp32.shape[0] < TINY_THRESHOLD:
                            optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_TINY)
                        elif X_train_fp32.shape[0] < SMALL_THRESHOLD:
                            optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_SMALL)
                        elif X_train_fp32.shape[0] < MEDIUM_THRESHOLD:
                            optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_MEDIUM)
                        elif X_train_fp32.shape[0] < LARGE_THRESHOLD:
                            optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_LARGE)
                        else:
                            optimal_batch_size = min(optimal_batch_size, BATCH_SIZE_HUGE)
                        
                        # Ensure batch size is reasonable relative to dataset size
                        optimal_batch_size = min(optimal_batch_size, len(X_train_fp32))
                        
                        # Update batch size
                        old_batch_size = batch_size
                        batch_size = optimal_batch_size
                        
                        # Check if batch size changed significantly (more than 10%)
                        if abs(old_batch_size - batch_size) / old_batch_size > 0.1:
                            batch_size_ratio = batch_size / old_batch_size
                            old_lr = optimizer.param_groups[0]['lr']
                            new_lr = old_lr * (batch_size_ratio ** BATCH_POWER)
                            
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            
                            print(f"[GPU {gpu_id}] Model {i}: Updated fine-tuning batch size from {old_batch_size} to {batch_size}")
                            print(f"[GPU {gpu_id}] Model {i}: Updated fine-tuning LR from {old_lr:.6f} to {new_lr:.6f}")
                        else:
                            print(f"[GPU {gpu_id}] Model {i}: Fine-tuning batch size change too small, keeping at {batch_size}")
                        
                        ft_gns_measured_epochs.append(ft_epoch)
                
                model.train()
                
                # For small datasets, use full batch for perfect memorization
                if len(X_train_fp32) <= 100:
                    batch_X, batch_y = X_train_fp32, y_train_fp32
                else:
                    batch_indices = torch.randperm(len(X_train_fp32), device=device)[:batch_size]
                    batch_X = X_train_fp32[batch_indices]
                    batch_y = y_train_fp32[batch_indices]
                
                # One training step with full precision
                optimizer.zero_grad(set_to_none=True)
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
                
                loss.backward()
                optimizer.step()
                
                # Step the scheduler after warmup
                if ft_epoch >= ft_warmup_epochs:
                    # For first epoch after warmup, reset scheduler
                    if ft_epoch == ft_warmup_epochs:
                        ft_current_lr = optimizer.param_groups[0]['lr']
                        ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=fine_tuning_epochs - ft_warmup_epochs,
                            eta_min=ft_current_lr / 100
                        )
                        print(f"[GPU {gpu_id}] Model {i}: Fine-tuning warmup complete, starting scheduler at LR={ft_current_lr:.8f}")
                    
                    ft_scheduler.step()
                
                # Evaluate at end or periodically
                if ft_epoch == fine_tuning_epochs - 1 or (ft_epoch > 0 and ft_epoch % 100 == 0):
                    with torch.no_grad():
                        model.eval()
                        eval_output = model(X_eval_fp32)
                        
                        X_test_fp32 = X_test.to(dtype=torch.float32)
                        y_test_fp32 = y_test.to(dtype=torch.float32)
                        test_output = model(X_test_fp32)
                        
                        train_error = torch.mean((eval_output - y_eval_fp32) ** 2).item()
                        test_error = torch.mean((test_output - y_test_fp32) ** 2).item()
                        
                        # Only record at very end
                        if ft_epoch == fine_tuning_epochs - 1:
                            train_errors[i].append(train_error)
                            test_errors[i].append(test_error)
                            epoch_numbers[i].append(max_epochs + ft_epoch + 1)
            
            # Store final batch size from fine-tuning
            batch_sizes[i] = batch_size
            
            # Restore original dtype
            model.to(dtype=model_dtype)
    
    # Save results
    completed_count = 0
    for i in range(len(models)):
        model = models[i]
        config_item = config_items[i]
        X_train, y_train = train_data[i]
        unique_id = unique_ids[i]
        
        # Final evaluation - use FULL training set for final result
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                model.eval()
                
                train_output = model(X_train)
                test_output = model(X_test)
                
                final_train_error = torch.mean((train_output - y_train) ** 2).item()
                final_test_error = torch.mean((test_output - y_test) ** 2).item()
        
        # Add final epoch if not already added
        if epoch_numbers[i][-1] != max_epochs + fine_tuning_epochs:
            train_errors[i].append(final_train_error)
            test_errors[i].append(final_test_error)
            epoch_numbers[i].append(max_epochs + fine_tuning_epochs)
        
        # Get final learning rate and batch size
        final_lr = optimizers[i].param_groups[0]['lr']
        final_batch_size = batch_sizes[i]
        
        # Get experiment details - architecture, k-factor, and initialization variance
        architecture = config_item.get('architecture', [])
        arch_str = 'x'.join([str(dim) for dim in architecture]) if architecture else ""
        init_variance = config_item.get('init_variance', 0.1)
        k_factor = config_item.get('k_factor')
        ds_depth = config_item.get('ds_depth')
        
        # Build result dictionary with k-factor information
        result = {
            'dataset_name': config_item['ds_name'],
            'dataset_directory': config_item['ds_directory'],
            'hidden_size': config_item['hidden_size'],
            'depth': config_item['depth'],
            'input_dim': config_item['input_dim'],
            'base_width': config_item.get('base_width', 10),
            'n_train': config_item['n_train'],
            'learning_rate': config_item['lr'],
            'mode': config_item['mode'],
            'alignment': config_item['alignment'],
            'distribution_type': 'gaussian',
            'architecture': architecture,
            'architecture_str': arch_str,
            'init_variance': init_variance,
            'normalize_data': config_item.get('normalize_data', False),
            'standardize_data': config_item.get('standardize_data', False),
            'ds_depth': ds_depth,  # Target network depth (new for k-factor)
            'k_factor': k_factor,  # Growth factor (new for k-factor)
            'test_error': final_test_error,
            'initial_train_error': train_errors[i][0],
            'final_train_error': final_train_error,
            'error_history': {
                'train_errors': train_errors[i],
                'test_errors': test_errors[i],
                'epochs': epoch_numbers[i],
                'early_stopped': early_stop_flags[i],
                'stopped_epoch': epoch_numbers[i][-1],
                'best_error': best_errors[i],
                'final_lr': final_lr,
                'eval_subset_size': min(EVAL_SUBSET_SIZE, len(X_train)),
                'final_batch_size': final_batch_size,
                'used_fp32_memorization': True,
                'ft_min_lr': optimizers[i].param_groups[0]['lr'] / 1000,
                'gns_measured_epochs': gns_measured_epochs[i],
                'used_extended_warmup': True,
                'warmup_epochs': warmup_epochs,
                'ft_warmup_epochs': ft_warmup_epochs
            },
            'worker_gpu': gpu_id,
            'model_seed': hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}"),
            'experiment_num': config_item['experiment_num'],
            'sweep_name': config_item['sweep_name'],
            'parallel_trained': True,
            'initial_batch_size': initial_batch_size,
            'optimal_batch_size': final_batch_size,
            'batch_power': BATCH_POWER,
            'multiple_gns_measurements': True
        }
        
        # Save results
        results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
        with open(results_file_path, "a") as f:
            # Convert any tensors to JSON-serializable types
            json_safe_result = convert_tensors_for_json(result)
            f.write(json.dumps(json_safe_result) + "\n")
            f.flush()
        
        # Save final model if requested
        if base_config.get('save_model', False):
            final_model_dir = os.path.join(full_results_dir, "final_models")
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, f"final_model_{unique_id}.pt")
            save_model(model, final_model_path)
        
        # Mark as completed
        with open(checkpoint_log_path, "a") as cp_f:
            cp_f.write(unique_id + "\n")
        completed_configs.add(unique_id)
        completed_count += 1
    
    return completed_count


def main():
    try:
        start_time = time.time()
        print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        if len(sys.argv) < 2:
            print("Usage: python main.py <config_file.yaml>")
            sys.exit(1)
        
        config_path = sys.argv[1]
        print(f"Loading config from: {config_path}")
        
        config = load_yaml_config(config_path)
        
        # Extract Base Config
        base_cfg = config["base_config"]
        base_results_dir = base_cfg["base_results_dir"]
        restart_checkpoint = base_cfg.get("restart_checkpoint")
        
        # Ensure fine_tuning_epochs is set
        if "fine_tuning_epochs" not in base_cfg:
            base_cfg["fine_tuning_epochs"] = 500
        
        # Create experiment name
        sweep_names = list(config["sweeps"].keys())
        experiment_name = f"{'_'.join(sweep_names)}_exp_{datetime.now().strftime('%Y%m%d')}"
        
        # Set up Results Directory
        full_results_dir = os.path.join(base_results_dir, experiment_name)
        os.makedirs(full_results_dir, exist_ok=True)
        
        # Set up Checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_log_path = os.path.join(full_results_dir, f"checkpoint_{timestamp}.txt")
        
        # Handle restart logic
        if restart_checkpoint is not None:
            checkpoint_log_path = restart_checkpoint
            with open(checkpoint_log_path, "r") as f:
                completed_configs = set(line.strip() for line in f if line.strip())
            timestamp = os.path.basename(restart_checkpoint).replace("checkpoint_", "").replace(".txt", "")
            print(f"Restarting from checkpoint with {len(completed_configs)} completed configurations")
        else:
            if os.path.exists(checkpoint_log_path):
                with open(checkpoint_log_path, "r") as f:
                    completed_configs = set(line.strip() for line in f if line.strip())
                print(f"Using existing checkpoint with {len(completed_configs)} completed configs")
            else:
                completed_configs = set()
                print(f"Starting new run")
            
            # Save hyperparameters
            hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.yaml")
            with open(hyperparams_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Generate all combinations
        print("Generating parameter combinations...")
        all_combinations = generate_all_combinations(config)
        print(f"Generated {len(all_combinations)} combinations")
        
        # Filter out completed configurations
        remaining = [c for c in all_combinations if generate_unique_id(c) not in completed_configs]
        print(f"Remaining configurations to process: {len(remaining)}/{len(all_combinations)}")
        
        if not remaining:
            print("All configurations already completed!")
            return
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available. Running on CPU.")
            num_gpus = 1
        
        print(f"Using {num_gpus} GPU(s)")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            print("spawn method already set")
        
        # Launch one process per GPU
        mp.spawn(
            worker_process,
            args=(num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs),
            nprocs=num_gpus,
            join=True
        )
        
        total_time = time.time() - start_time
        print(f"All processes completed in {total_time:.2f} seconds")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()