import numpy as np
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpi4py import MPI
import gzip
import io
import json
import time
import math
from itertools import combinations_with_replacement
import random
import traceback
import multiprocessing
from threading import Thread
from queue import Queue
from torch.nn import functional as F
import scipy.linalg

def save_dataset_compressed(X, y, filepath, rank, max_retries=3):
    """
    Save full dataset with compression, with verification and retries.
    Uses a separate process for compression to offload from main thread.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Process {rank}: Directory verified: {directory}")
    except Exception as e:
        print(f"Process {rank}: ERROR creating directory {directory}: {e}")
        return False
    
    def compression_worker(data_queue, result_queue):
        """Worker function to handle compression in a separate process"""
        try:
            X_f32, y_f32, filepath = data_queue.get()
            
            # Create a buffer to compress the data
            buffer = io.BytesIO()
            torch.save({'X': X_f32, 'y': y_f32}, buffer)
            compressed_data = gzip.compress(buffer.getvalue(), compresslevel=9)
            
            # Save compressed data
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to flush file buffers
            
            # Verify the file exists with non-zero size
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                file_size_mb = file_size / (1024 * 1024)
                result_queue.put((True, file_size_mb))
            else:
                result_queue.put((False, 0))
        except Exception as e:
            result_queue.put((False, str(e)))
    
    # Try multiple times to save
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Process {rank}: Compressing dataset (attempt {attempt}/{max_retries})...")
            
            # Convert to float32 to save space and move to CPU
            if X is not None:
                X_f32 = X.detach().cpu().to(torch.float32)
            else:
                X_f32 = None
                
            if y is not None:
                y_f32 = y.detach().cpu().to(torch.float32)
            else:
                y_f32 = None
            
            # Setup queues for inter-process communication
            data_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            
            # Start compression worker process
            data_queue.put((X_f32, y_f32, filepath))
            proc = multiprocessing.Process(target=compression_worker, args=(data_queue, result_queue))
            proc.start()
            
            # Wait for result with timeout
            proc.join(timeout=300)  # 5 minutes timeout
            
            # Check if process is still alive (timeout occurred)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                print(f"Process {rank}: Compression timeout, retrying...")
                continue
            
            # Get result from queue
            if not result_queue.empty():
                success, result = result_queue.get()
                if success:
                    print(f"Process {rank}: VERIFIED: Saved compressed dataset ({result:.2f} MB) to {filepath}")
                    return True
                else:
                    print(f"Process {rank}: WARNING: Compression failed: {result}")
            else:
                print(f"Process {rank}: WARNING: No result from compression worker")
                
        except Exception as e:
            print(f"Process {rank}: ERROR during save attempt {attempt}: {e}")
            traceback.print_exc()
            
        # Only retry if not the last attempt
        if attempt < max_retries:
            print(f"Process {rank}: Waiting before retry...")
            time.sleep(2)  # Wait before retrying
        else:
            print(f"Process {rank}: FAILED: Could not save dataset after {max_retries} attempts")
            return False
    
    return False

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def save_results(results, save_dir, name, max_retries=3):
    """Save results with verification and retries."""
    # Create directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory verified: {save_dir}")
    except Exception as e:
        print(f"ERROR creating directory {save_dir}: {e}")
        return False
    
    # Create filename
    filepath = os.path.join(save_dir, f"results_{name}.json")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Saving results (attempt {attempt}/{max_retries})...")
            
            # Use temporary file approach for safety
            temp_filepath = filepath + ".tmp"
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = convert_numpy_types(results)
            
            # Convert to JSON format with pretty print
            with open(temp_filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify temp file exists with content
            if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 0:
                # Verify JSON is valid by reading it back
                with open(temp_filepath, 'r') as f:
                    # Just try to load it to verify integrity
                    json.load(f)
                
                # If we get here, JSON is valid, so rename to final file
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
                
                print(f"VERIFIED: Saved results to {filepath}")
                return True
            else:
                print(f"WARNING: Results file was not created properly")
                
        except Exception as e:
            print(f"ERROR during results save attempt {attempt}: {e}")
            traceback.print_exc()
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
        # Only retry if not the last attempt
        if attempt < max_retries:
            print(f"Waiting before retry...")
            time.sleep(2)
        else:
            print(f"FAILED: Could not save results after {max_retries} attempts")
            return False
    
    return False

def generate_gaussian_data(train_size, d, device, batch_size=None):
    """
    Generate Gaussian data with mean 0 and variance 1.
    
    Args:
        train_size: Total number of samples to generate
        d: Input dimension
        device: GPU device to use
        batch_size: Batch size for generation (if None, will auto-select based on d)
    """
    # Auto-select batch size based on dimension if not provided
    if batch_size is None:
        if d <= 32:
            batch_size = 500000
        elif d <= 64:
            batch_size = 200000
        elif d <= 128:
            batch_size = 100000
        else:
            batch_size = 50000
    
    # Generate data in batches
    result_list = []
    remaining = train_size
    
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        
        # Generate Gaussian random vectors
        batch = torch.randn(current_batch_size, d, device=device, dtype=torch.float32)
        
        result_list.append(batch)
        remaining -= current_batch_size
    
    # Concatenate batches
    result = torch.cat(result_list, dim=0)
    return result

class NeuralNetworkTargetFunction(torch.nn.Module):
    """
    Neural network target function with configurable architecture.
    """
    def __init__(self, input_dim, hidden_layers, init_variance, device):
        """
        Initialize neural network with specified architecture.
        
        Args:
            input_dim: Input dimension
            hidden_layers: List of hidden layer dimensions
            init_variance: Variance for weight initialization
            device: Device to place the model on
        """
        super(NeuralNetworkTargetFunction, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.init_variance = init_variance
        self.device = device
        
        # Create list to store layers and activation functions
        self.layers = torch.nn.ModuleList()
        
        # Keep track of dimensions for each layer (including input and output)
        self.layer_dims = [input_dim] + hidden_layers + [1]
        
        # Create layers
        for i in range(len(self.layer_dims) - 1):
            # Create linear layer without bias
            layer = torch.nn.Linear(self.layer_dims[i], self.layer_dims[i+1], bias=False)
            
            # Initialize weights according to He initialization (standard for ReLU)
            # but scaled by the specified variance factor
            std = math.sqrt(2.0 / self.layer_dims[i]) * math.sqrt(init_variance)
            torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
            
            self.layers.append(layer)
        
        # Move model to device
        self.to(device)
        print(f"Created NeuralNetworkTargetFunction with architecture {self.layer_dims} on {device}")
        print(f"Weight initialization variance: {init_variance}")
    
    def forward(self, x, return_all_activations=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            return_all_activations: If True, return activations from all hidden layers
        
        Returns:
            Network output, and optionally all hidden activations
        """
        activations = []
        h = x
        
        # Pass through all but the last layer with ReLU activation
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            activations.append(h.clone())  # Store pre-activation for kernel computation
            h = F.relu(h)
        
        # Last layer (no activation)
        output = self.layers[-1](h)
        
        if return_all_activations:
            return output, activations
        else:
            return output
    
    def compute_kernel_matrices(self, X, batch_size=None):
        """
        Compute conjugate kernel matrices h^T(x)h(x) for each hidden layer using all data.
        The kernel is computed by summing over all samples and dividing by N.
        
        Args:
            X: All input data of shape [N, input_dim]
            batch_size: Batch size for memory-efficient computation
        
        Returns:
            Dictionary containing kernel matrices, ranks, and eigenvalues for each layer
        """
        # Initialize to hold kernel accumulation for each layer
        kernel_accumulators = None
        total_samples = 0
        
        # Determine batch size if not provided
        if batch_size is None:
            if self.input_dim <= 32:
                batch_size = 50000
            elif self.input_dim <= 64:
                batch_size = 25000
            elif self.input_dim <= 128:
                batch_size = 10000
            else:
                batch_size = 5000
                
            # Adjust further based on max hidden layer size
            max_hidden = max(self.hidden_layers) if self.hidden_layers else 0
            if max_hidden >= 128:
                batch_size = max(1000, batch_size // 2)
        
        # Process data in batches
        num_batches = (len(X) + batch_size - 1) // batch_size
        print(f"Computing kernel matrices using all {len(X)} samples in {num_batches} batches")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch_size_actual = end_idx - start_idx
            
            # Get activations for this batch
            with torch.no_grad():
                _, activations = self.forward(X[start_idx:end_idx], return_all_activations=True)
            
            # If first batch, initialize accumulators
            if kernel_accumulators is None:
                kernel_accumulators = []
                for h in activations:
                    h_cpu = h.cpu().numpy()
                    # Initialize accumulator with first batch contribution
                    kernel = np.matmul(h_cpu.T, h_cpu)
                    kernel_accumulators.append(kernel)
            else:
                # For subsequent batches, accumulate contributions
                for j, h in enumerate(activations):
                    h_cpu = h.cpu().numpy()
                    kernel_accumulators[j] += np.matmul(h_cpu.T, h_cpu)
            
            total_samples += batch_size_actual
            
            # Report progress
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Processed {i+1}/{num_batches} batches ({total_samples}/{len(X)} samples)")
                
            # Clear GPU memory
            del activations
            torch.cuda.empty_cache()
        
        # Now process the accumulated kernels
        results = {}
        
        # Normalize by dividing by total samples and compute eigenvalues/ranks
        for i, kernel_matrix in enumerate(kernel_accumulators):
            layer_name = f"layer_{i+1}"
            
            # Normalize by total samples
            kernel_matrix /= total_samples
            
            # Calculate eigenvalues
            eigenvalues, _ = np.linalg.eigh(kernel_matrix)
            eigenvalues = eigenvalues.tolist()
            
            # Calculate numerical rank (using SVD)
            # Use scipy for better numerical stability
            s = scipy.linalg.svdvals(kernel_matrix)
            rank = np.sum(s > 1e-10)  # Count singular values above threshold
            
            hidden_dim = self.hidden_layers[i]
            results[layer_name] = {
                "hidden_dim": hidden_dim,
                "rank": int(rank),
                "eigenvalues": eigenvalues,
                "matrix_shape": [hidden_dim, hidden_dim]
            }
        
        return results

def generate_batch_efficiently(model, X_batch, device):
    """
    Generate target values for a batch of inputs efficiently.
    """
    try:
        with torch.no_grad():
            y_batch = model(X_batch)
        return y_batch
    except RuntimeError as e:
        # If we get a memory error, try with smaller sub-batches
        if "out of memory" in str(e).lower():
            print(f"Memory error with batch size {len(X_batch)}, trying smaller batches")
            torch.cuda.empty_cache()
            
            sub_batch_size = len(X_batch) // 2
            y_parts = []
            
            for i in range(0, len(X_batch), sub_batch_size):
                end_idx = min(i + sub_batch_size, len(X_batch))
                with torch.no_grad():
                    y_part = model(X_batch[i:end_idx])
                y_parts.append(y_part)
            
            return torch.cat(y_parts, dim=0)
        else:
            raise

def format_architecture(hidden_layers):
    """Format the architecture for filenames"""
    return "x".join(str(h) for h in hidden_layers)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get total available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Set up GPUs - assuming we have 4 of them as specified
    num_h100s = 4
    
    # Assign specific GPU to this MPI process
    # Ensure we distribute across all 4 GPUs
    assigned_gpu = rank % num_h100s
    
    # Set default device for this process to the assigned GPU
    torch.cuda.set_device(assigned_gpu)
    device = torch.device(f'cuda:{assigned_gpu}')
    
    if rank == 0:
        print(f"Number of available GPUs: {num_gpus}")
        print(f"Using {num_h100s} GPUs for computation")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} with {props.total_memory / 1e9:.1f} GB memory")
    
    print(f"Process {rank}/{size} assigned to GPU {assigned_gpu}, device = {device}")
    comm.Barrier()  # Synchronize for clean output
    
    # --- Hyperparameters ---
    # List of input dimensions to explore
    input_dimensions = [16, 32, 64, 128, 256]
    
    # List of neural network architectures to explore
    # Each entry is a list of hidden layer dimensions
    architectures = [
        [16, 8, 4, 2],
        [16, 16, 16, 16],
        [16, 32, 64, 128],
        [32, 16, 8, 4],
        [32, 32, 32, 32],
        [32, 64, 128, 256]
    ]
    
    # List of initialization variances to explore
    init_variances = [0.1, 0.5, 1.0, 2.0]
    
    # Training samples - adjusted for efficiency
    train_size = 5100000
    
    # Number of experiments (different random initializations)
    num_experiments = 1
    
    # Base directory for saving data
    data_base_dir = "/scratch/goring/NN_data/NN_0304"
    
    # Calculate total number of combinations
    total_combinations = []
    
    for input_dim in input_dimensions:
        for architecture in architectures:
            for variance in init_variances:
                for exp_num in range(1, num_experiments + 1):
                    total_combinations.append((input_dim, architecture, variance, exp_num))
    
    # Distribute across MPI processes
    num_combinations = len(total_combinations)
    combinations_per_process = (num_combinations + size - 1) // size
    start_idx = rank * combinations_per_process
    end_idx = min((rank + 1) * combinations_per_process, num_combinations)
    
    # Process only the combinations assigned to this rank
    my_combinations = total_combinations[start_idx:end_idx]
    
    if rank == 0:
        print(f"\nTotal combinations: {num_combinations}")
        print(f"Number of MPI processes: {size}")
        print(f"Data base directory: {data_base_dir}")
    
    # Display this process's assignment
    print(f"Process {rank}: Processing {len(my_combinations)} combinations from {start_idx} to {end_idx-1}")
    comm.Barrier()  # Synchronize for clean output
    
    # Process each combination assigned to this rank
    for idx, (input_dim, architecture, variance, exp_num) in enumerate(my_combinations):
        # Create a compact name for this combination
        arch_str = format_architecture(architecture)
        var_str = f"var{variance}".replace(".", "p")
        run_name = f"NN_gaussian_d{input_dim}_arch{arch_str}_{var_str}"
        
        print(f"\nProcess {rank} starting combination {idx+1}/{len(my_combinations)}: {run_name} (Exp {exp_num})")
        print(f"Experiment {exp_num}/{num_experiments} - different NN initialization with same architecture")
        
        # Set seed based on experiment number and rank for uniqueness
        seed = 42 + exp_num + rank * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            # Initialize GPU to ensure it's ready
            torch.cuda.empty_cache()
            
            # Generate Gaussian data
            print(f"Process {rank} generating Gaussian data on GPU {assigned_gpu}")
            
            # Set batch size based on dimension
            if input_dim <= 32:
                gen_batch_size = 500000
            elif input_dim <= 64:
                gen_batch_size = 300000
            elif input_dim <= 128:
                gen_batch_size = 150000
            else:
                gen_batch_size = 75000
                
            X = generate_gaussian_data(train_size, input_dim, device, batch_size=gen_batch_size)
            
            print(f"Process {rank}: Data generated with shape {X.shape}")
            
            # Create neural network model
            print(f"Process {rank} creating neural network on GPU {assigned_gpu}")
            model = NeuralNetworkTargetFunction(input_dim, architecture, variance, device)
            
            # Process data - use efficient batching for high dimensions
            print(f"Process {rank} computing target function with {X.shape[0]} samples")
            
            # Determine optimal batch size based on dimension and network size
            if input_dim <= 32:
                batch_size = 100000
            elif input_dim <= 64:
                batch_size = 75000
            elif input_dim <= 128:
                batch_size = 50000
            else:
                batch_size = 25000
            
            # Adjust batch size based on network size (largest hidden layer)
            largest_hidden = max(architecture) if architecture else 0
            if largest_hidden >= 128:
                batch_size = max(10000, batch_size // 2)
            
            # Generate results in batches
            num_batches = (X.shape[0] + batch_size - 1) // batch_size
            
            # Track progress
            start_time = time.time()
            y_list = []
            
            for i in range(num_batches):
                start_idx_batch = i * batch_size
                end_idx_batch = min((i + 1) * batch_size, X.shape[0])
                
                batch_size_actual = end_idx_batch - start_idx_batch
                print(f"Process {rank}: Processing batch {i+1}/{num_batches} ({start_idx_batch}-{end_idx_batch}, {batch_size_actual} samples)")
                
                # Generate target values for this batch
                try:
                    y_batch = generate_batch_efficiently(model, X[start_idx_batch:end_idx_batch], device)
                    y_list.append(y_batch)
                    
                    # Report progress and timing
                    if i > 0:
                        elapsed = time.time() - start_time
                        samples_processed = (i + 1) * batch_size
                        throughput = samples_processed / elapsed
                        eta = (num_batches - i - 1) * elapsed / (i + 1)
                        print(f"Process {rank}: Progress {(i+1)/num_batches:.1%}, Throughput: {throughput:.2f} samples/sec, ETA: {eta:.1f} sec")
                        
                except Exception as e:
                    print(f"Process {rank}: Error processing batch {i+1}: {e}")
                    traceback.print_exc()
                    # Try with a smaller batch size
                    sub_size = batch_size_actual // 2
                    print(f"Process {rank}: Retrying with smaller batch size {sub_size}")
                    
                    for j in range(start_idx_batch, end_idx_batch, sub_size):
                        sub_end = min(j + sub_size, end_idx_batch)
                        sub_y = generate_batch_efficiently(model, X[j:sub_end], device)
                        y_list.append(sub_y)
                
                # Periodically clear cache to avoid memory buildup
                torch.cuda.empty_cache()
            
            # Concatenate results
            y = torch.cat(y_list, dim=0)
            del y_list  # Free memory
            
            # Verify data integrity
            print(f"Process {rank}: Generated y values with shape {y.shape}")
            print(f"Process {rank}: X shape: {X.shape}, Y shape: {y.shape}")
            if len(X) != len(y):
                raise ValueError(f"Length mismatch: X has {len(X)} samples but y has {len(y)} samples")
            
            # Compute kernel matrices using all data
            print(f"Process {rank}: Computing conjugate kernel matrices (h^T(x)h(x)) using all data")
            kernel_results = model.compute_kernel_matrices(X)
            print(f"Process {rank}: Kernel computation complete")
            
            # Save all results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            smart_name = f"NN_gaussian_d{input_dim}_arch{arch_str}_{var_str}_{exp_num}"
            data_subdir = f"NN_{smart_name}_{timestamp}"
            data_save_dir = os.path.join(data_base_dir, data_subdir)
            
            try:
                os.makedirs(data_save_dir, exist_ok=True)
                
                # Save metadata and results
                results = {
                    'hyperparameters': {
                        'distribution_type': 'gaussian',
                        'input_dim': int(input_dim),
                        'architecture': [int(h) for h in architecture],
                        'init_variance': float(variance),
                        'train_size': int(train_size),
                        'experiment_number': int(exp_num),
                        'experiment_description': f"Random NN #{exp_num} with same architecture",
                        'random_seed': int(seed),
                        'mpi_rank': int(rank),
                        'mpi_size': int(size),
                        'gpu_device': int(assigned_gpu)
                    },
                    'neural_network': {
                        'layer_dimensions': [int(d) for d in model.layer_dims],
                        'activation': 'ReLU',
                        'bias': False,
                        'init_type': 'He (PyTorch default for ReLU)',
                        'init_variance': float(variance)
                    },
                    'kernel_analysis': kernel_results
                }
                
                save_results(results, data_save_dir, smart_name)
                
                # Save dataset - each process saves its own results
                x_path = os.path.join(data_save_dir, f"dataset_X_{smart_name}.pt.gz")
                y_path = os.path.join(data_save_dir, f"dataset_y_{smart_name}.pt.gz")
                
                save_dataset_compressed(X, None, x_path, rank)
                save_dataset_compressed(None, y, y_path, rank)
                
                print(f"Process {rank}: Successfully saved data for {run_name}")
                
            except Exception as e:
                print(f"Process {rank}: ERROR during save: {e}")
                traceback.print_exc()
            
            # Clean up to save memory before next combination
            del X, y, model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Process {rank}: ERROR processing {run_name}: {e}")
            traceback.print_exc()
    
    # Final synchronization
    comm.Barrier()
    if rank == 0:
        print("\nAll processes completed their assigned work.")

if __name__ == "__main__":
    main()