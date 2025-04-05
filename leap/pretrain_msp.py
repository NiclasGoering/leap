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

def save_dataset_compressed(X, y, filepath, rank, max_retries=3):
    """
    Save full dataset with compression, with verification and retries.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Process {rank}: Directory verified: {directory}")
    except Exception as e:
        print(f"Process {rank}: ERROR creating directory {directory}: {e}")
        return False
    
    # Try multiple times to save
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Process {rank}: Compressing dataset (attempt {attempt}/{max_retries})...")
            
            # Convert to float32 to save space
            if X is not None:
                X_f32 = X.detach().cpu().to(torch.float32)
            else:
                X_f32 = None
                
            if y is not None:
                y_f32 = y.detach().cpu().to(torch.float32)
            else:
                y_f32 = None
            
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
                
                if file_size > 0:
                    print(f"Process {rank}: VERIFIED: Saved compressed dataset ({file_size_mb:.2f} MB) to {filepath}")
                    return True
                else:
                    print(f"Process {rank}: WARNING: File has zero size, retrying...")
            else:
                print(f"Process {rank}: WARNING: File was not created, retrying...")
                
        except Exception as e:
            print(f"Process {rank}: ERROR during save attempt {attempt}: {e}")
            
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

def generate_data(distribution_type, train_size, d, device, batch_size=None):
    """
    Generate data from specified distribution in batches for memory efficiency.
    
    Args:
        distribution_type: Type of distribution ('boolean_cube')
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
        
        if distribution_type == 'boolean_cube':
            # Generate uniform random binary vectors in {-1, 1}^d
            batch = 2 * torch.randint(0, 2, (current_batch_size, d), device=device, dtype=torch.float32) - 1
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        result_list.append(batch)
        remaining -= current_batch_size
    
    # Concatenate batches
    result = torch.cat(result_list, dim=0)
    return result

def improved_generate_msp_structure(d, max_degree, leap):
    """
    Enhanced algorithm to generate MSP structure with the exact leap.
    Completely randomizes the variable selection for each monomial.
    
    Args:
        d: Input dimension
        max_degree: Maximum degree of monomials
        leap: Exact leap between monomials
        
    Returns:
        A list of sets representing the monomials.
    """
    # Make sure leap is within bounds
    leap = min(max_degree, leap)
    
    # Initialize with constant term
    monomials = [set()]
    
    # Shuffle all variables to randomize selection
    all_variables = list(range(d))
    random.shuffle(all_variables)
    
    # Start with first variable monomial
    if d >= 1:
        first_var = all_variables[0]
        monomials.append({first_var})
    
    # Keep track of used variables
    used_vars = {first_var} if d >= 1 else set()
    
    # Add monomials with increasing degrees up to max_degree
    current_degree = 1
    
    # Try to add more monomials until we reach max degree
    while current_degree < max_degree and len(all_variables) > len(used_vars):
        # Calculate how many variables we need for the next monomial
        # We need exactly leap new variables
        remaining_vars = list(set(all_variables) - used_vars)
        
        if len(remaining_vars) < leap:
            # Not enough remaining variables
            break
            
        # Shuffle remaining variables for randomness
        random.shuffle(remaining_vars)
        
        # Select leap random variables
        new_vars = set(remaining_vars[:leap])
        
        # Decide whether to include some previous variables or create disjoint set
        include_previous = random.choice([True, False])
        
        if include_previous and used_vars:
            # Include some previous variables (randomly selected)
            prev_vars_list = list(used_vars)
            random.shuffle(prev_vars_list)
            # Take up to max_degree - leap previous variables
            num_prev = min(random.randint(1, len(prev_vars_list)), max_degree - leap)
            prev_vars = set(prev_vars_list[:num_prev])
            
            new_monomial = new_vars.union(prev_vars)
        else:
            # Just use the new variables
            new_monomial = new_vars
        
        # Make sure we don't exceed max_degree
        if len(new_monomial) <= max_degree:
            monomials.append(new_monomial)
            used_vars.update(new_vars)
            current_degree = max(current_degree, len(new_monomial))
    
    # Sort monomials by degree (size) for better handling
    monomials.sort(key=len)
    
    # Verify the structure has the correct leap
    actual_leap = get_leap(monomials)
    
    # If the leap doesn't match the expected value, try to fix it
    if actual_leap != leap and leap > 0:
        print(f"Warning: Generated structure has leap={actual_leap}, expected={leap}. Regenerating...")
        return create_exact_leap_structure(leap, max_degree, d)
    
    return monomials

def create_exact_leap_structure(leap, max_degree, d):
    """
    Create a minimal structure with exactly the specified leap using random variables.
    """
    if leap == 0:
        return [set()]
    
    if leap > d:
        leap = d  # Can't have leap larger than dimension
    
    # Shuffle all variables to randomize selection
    all_variables = list(range(d))
    random.shuffle(all_variables)
    
    # Create a very simple structure with exactly the specified leap
    monomials = [
        set(),  # Constant term
        {all_variables[0]},  # First variable, randomly selected
    ]
    
    # For the third monomial, select exactly leap random variables
    # Ensure they're different from the first variable
    leap_vars = set(all_variables[1:leap+1])
    
    # Sometimes include the first variable
    include_first = random.choice([True, False])
    
    if include_first:
        third_monomial = {all_variables[0]}.union(leap_vars)
    else:
        third_monomial = leap_vars
    
    # Add the third monomial if it doesn't exceed max degree
    if len(third_monomial) <= max_degree:
        monomials.append(third_monomial)
    
    return monomials

def get_leap(monomials):
    """
    Calculate the actual leap of a monomial structure.
    Returns the maximum minimum leap across all monomials.
    """
    if len(monomials) <= 1:
        return 0
        
    sorted_monomials = sorted(monomials, key=len)
    max_min_leap = 0
    
    for i in range(1, len(sorted_monomials)):
        # Find smallest leap from any prior monomial
        min_leap = float('inf')
        for j in range(i):
            leap = len(sorted_monomials[i] - sorted_monomials[j])
            min_leap = min(min_leap, leap)
        max_min_leap = max(max_min_leap, min_leap)
            
    return max_min_leap

def verify_msp_structure(monomials, expected_leap):
    """
    Verify that the MSP structure has the expected leap property.
    """
    if not monomials:
        return True
    
    actual_leap = get_leap(monomials)
    if actual_leap != expected_leap:
        print(f"Leap verification failed: expected={expected_leap}, actual={actual_leap}")
        return False
    
    return True

def optimized_compute_monomial_torch(x, monomial_indices):
    """
    Optimized function to compute the product of variables.
    For different monomial sizes, uses different computational strategies:
    - Empty monomial: return ones
    - Single variable: direct indexing
    - Small monomials (<=8): direct product
    - Large monomials: batched approach for memory efficiency
    
    Args:
        x: Input tensor of shape [batch_size, d]
        monomial_indices: Indices of variables to multiply
    Returns:
        Monomial values for each input
    """
    if not monomial_indices:  # Empty set = constant 1
        return torch.ones(x.shape[0], device=x.device)
    
    # Special case for single-variable monomials (very common)
    if len(monomial_indices) == 1:
        idx = next(iter(monomial_indices))
        return x[:, idx].clone()
    
    # For few variables, direct product is more efficient
    if len(monomial_indices) <= 8:
        # Start with ones
        result = torch.ones(x.shape[0], device=x.device)
        # Multiply each variable
        for idx in monomial_indices:
            result *= x[:, idx]
        return result
    
    # For many variables, use the standard approach with a single tensor operation
    idx_list = list(monomial_indices)
    return torch.prod(x[:, idx_list], dim=1)

class EfficientStaircaseFunction:
    """
    Improved implementation of MSP function with all unit coefficients.
    """
    def __init__(self, msp_structure, d, device):
        self.msp_structure = msp_structure
        self.d = d
        self.device = device
        
        # Count total monomials
        self.total_monomials = len(msp_structure)
        
        # All coefficients are 1.0 - no randomization
        self.coefficients = torch.ones(self.total_monomials, device=device)
        
        # Group monomials by degree for easier processing
        self.monomials_by_degree = {}
        self.degree_counts = {}
        
        for i, monomial in enumerate(msp_structure):
            degree = len(monomial)
            if degree not in self.monomials_by_degree:
                self.monomials_by_degree[degree] = []
                self.degree_counts[degree] = 0
            self.monomials_by_degree[degree].append((i, monomial))
            self.degree_counts[degree] += 1
        
        print(f"Created EfficientStaircaseFunction with {self.total_monomials} monomials on {device}")
        print(f"Monomial degree distribution: {self.degree_counts}")
        print(f"All monomials have coefficient 1.0 (no random weights)")
        
    def get_max_degree(self):
        """Get the maximum degree of the function."""
        if not self.msp_structure:
            return 0
        return max(len(m) for m in self.msp_structure)
    
    def get_leap(self):
        """Calculate the actual leap of the structure."""
        return get_leap(self.msp_structure)
        
    def __call__(self, x, batch_size=None):
        """
        Compute the function value for each input in x, using batching for memory efficiency.
        
        Args:
            x: Input tensor of shape [batch_size, d]
            batch_size: Optional batch size for processing (auto-calculated if None)
            
        Returns:
            Function values as a tensor of shape [batch_size]
        """
        # Make sure input is on the right device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Determine appropriate batch size based on input size and dimension
        if batch_size is None:
            # Scale batch size inversely with dimension and number of monomials
            if self.d <= 32:
                batch_size = 100000
            elif self.d <= 64:
                batch_size = 50000
            elif self.d <= 128:
                batch_size = 20000
            else:
                batch_size = 10000
                
            # Adjust for number of monomials
            if self.total_monomials > 20:
                batch_size = max(5000, batch_size // 2)
        
        # Use batching for large inputs
        if len(x) > batch_size:
            result_batches = []
            for i in range(0, len(x), batch_size):
                end_idx = min(i + batch_size, len(x))
                batch_result = self._compute_batch(x[i:end_idx])
                result_batches.append(batch_result)
            return torch.cat(result_batches)
        else:
            return self._compute_batch(x)
    
    def _compute_batch(self, x_batch):
        """Compute function values for a single batch."""
        # Initialize result with zeros
        result = torch.zeros(x_batch.shape[0], device=self.device)
        
        # Process monomials grouped by degree for better memory efficiency
        for degree in sorted(self.monomials_by_degree.keys()):
            for idx, monomial in self.monomials_by_degree[degree]:
                # Compute monomial value using optimized function
                monomial_value = optimized_compute_monomial_torch(x_batch, monomial)
                
                # Since all coefficients are 1, simply add the monomial value
                result += monomial_value
                
            # Clear cache after processing all monomials of a degree (for high degrees)
            if degree > 10 and self.degree_counts[degree] > 5:
                torch.cuda.empty_cache()
        
        return result

def create_target_function(d, expected_leap, max_degree, device):
    """
    Create a target function with MSP structure and the expected leap.
    
    Args:
        d: Input dimension
        expected_leap: Expected leap of the function
        max_degree: Maximum degree to consider
        device: GPU device to use
    """
    # Generate MSP structure with improved algorithm
    msp_structure = improved_generate_msp_structure(d, max_degree, expected_leap)
    
    # Verify the structure has the correct leap
    actual_leap = get_leap(msp_structure)
    
    # Create the target function with our efficient implementation
    function = EfficientStaircaseFunction(msp_structure, d, device)
    
    # Print information about the function
    actual_max_degree = function.get_max_degree()
    
    print(f"Created target function with {len(msp_structure)} monomials")
    print(f"Expected leap: {expected_leap}, Actual leap: {actual_leap}")
    print(f"Expected max degree: {max_degree}, Actual max degree: {actual_max_degree}")
    print(f"MSP structure: {[list(m) for m in msp_structure]}")
    
    return function, msp_structure

def format_leap(leap):
    """Format leap for filename or display"""
    return f"leap{leap}"

def calculate_leap_values(max_degree):
    """
    Generate leap values according to the specified logic:
    1. Start with max_leap = max_degree
    2. Divide by 2 until reaching 5
    3. Then use leap values 5, 4, 3, 2, 1, 0
    """
    leap_values = []
    
    # Start with max_leap = max_degree
    leap = max_degree
    
    # Add max leap
    leap_values.append(leap)
    
    # Divide by 2 until reaching 5
    while leap > 5:
        leap = leap // 2
        if leap >= 5:
            leap_values.append(leap)
    
    # Add 5, 4, 3, 2, 1, 0 if not already included
    for l in [5, 4, 3, 2, 1, 0]:
        if l not in leap_values and l < max_degree:
            leap_values.append(l)
    
    return sorted(leap_values, reverse=True)

def calculate_degree_values(d, max_degree, num_points=10):
    """
    Calculate degree values for a given dimension d.
    Returns 10 points along the degree scaling up to max_degree.
    """
    # Ensure max_degree is at least 10 to get 10 points
    max_degree = max(max_degree, 10)
    
    # Generate num_points evenly spaced values
    degree_values = [max(1, round(i * max_degree / (num_points - 1))) for i in range(num_points)]
    
    # Remove duplicates while preserving order
    seen = set()
    degree_values = [x for x in degree_values if not (x in seen or seen.add(x))]
    
    return sorted(degree_values)

def get_max_degree_for_dimension(d, scaling_factor):
    """
    Calculate the maximum practical degree for a given dimension d.
    This is a heuristic to keep the computation manageable.
    """
    # Example scaling: d^0.5 would be a square root scaling
    max_degree = max(5, int(d ** scaling_factor))
    return min(max_degree, d)  # Cap at dimension d

def generate_batch_efficiently(function, X_batch, device):
    """
    Generate target values for a batch of inputs efficiently.
    """
    try:
        with torch.no_grad():
            y_batch = function(X_batch).unsqueeze(1)
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
                    y_part = function(X_batch[i:end_idx]).unsqueeze(1)
                y_parts.append(y_part)
            
            return torch.cat(y_parts, dim=0)
        else:
            raise

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get total available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Set up H100 GPUs - assuming we have 4 of them as specified
    num_h100s = 4
    
    # Assign specific GPU to this MPI process
    # Ensure we distribute across all 4 H100s
    assigned_gpu = rank % num_h100s
    
    # Set default device for this process to the assigned GPU
    torch.cuda.set_device(assigned_gpu)
    device = torch.device(f'cuda:{assigned_gpu}')
    
    if rank == 0:
        print(f"Number of available GPUs: {num_gpus}")
        print(f"Using {num_h100s} H100 GPUs for computation")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} with {props.total_memory / 1e9:.1f} GB memory")
    
    print(f"Process {rank}/{size} assigned to GPU {assigned_gpu}, device = {device}")
    comm.Barrier()  # Synchronize for clean output
    
    # --- Hyperparameters ---
    # Define distributions to explore - using boolean cube (binary vectors)
    distributions = ['boolean_cube']  # Uniform binary vectors in {-1, 1}^d
    
    # Fixed dimensions to explore (as specified)
    dimensions = [16, 32, 64, 128, 256]
    
    # Toggle for using degree scaling
    use_degree_scaling = False
    
    # List of scaling factors to try if use_degree_scaling is True
    scaling_factors = [0.4, 0.5, 0.6, 0.7]  # Different scaling approaches
    
    # Fixed maximum degree to use for all dimensions if use_degree_scaling is False
    fixed_max_degree = 14
    
    # Training samples - adjusted for efficiency
    train_size = 2100000
    
    # Number of experiments 
    num_experiments = 3
    
    # Base directory for saving data
    data_base_dir = "/scratch/goring/MSP_data/MSP_0104_fix"
    
    # Calculate total number of combinations
    total_combinations = []
    
    for dist in distributions:
        for dim in dimensions:
            if use_degree_scaling:
                # Use degree scaling based on dimension
                for scaling_factor in scaling_factors:
                    # Calculate max degree for this dimension and scaling factor
                    max_degree = get_max_degree_for_dimension(dim, scaling_factor)
                    
                    # Calculate degree values
                    degree_values = calculate_degree_values(dim, max_degree)
                    
                    for degree in degree_values:
                        # Calculate leap values
                        leap_values = calculate_leap_values(degree)
                        
                        for leap in leap_values:
                            for exp_num in range(1, num_experiments + 1):
                                total_combinations.append((dist, dim, degree, leap, scaling_factor, exp_num, True))
            else:
                # Use fixed max degree for all dimensions
                max_degree = min(fixed_max_degree, dim)  # Cap at dimension d
                
                # Calculate degree values
                degree_values = calculate_degree_values(dim, max_degree)
                
                for degree in degree_values:
                    # Calculate leap values
                    leap_values = calculate_leap_values(degree)
                    
                    for leap in leap_values:
                        for exp_num in range(1, num_experiments + 1):
                            total_combinations.append((dist, dim, degree, leap, None, exp_num, False))
    
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
        print(f"Using degree scaling: {use_degree_scaling}")
        if not use_degree_scaling:
            print(f"Fixed max degree: {fixed_max_degree}")
    
    # Display this process's assignment
    print(f"Process {rank}: Processing {len(my_combinations)} combinations from {start_idx} to {end_idx-1}")
    comm.Barrier()  # Synchronize for clean output
    
    # Process each combination assigned to this rank
    for idx, (dist_type, d, max_degree, leap, scaling_factor, exp_num, is_scaled) in enumerate(my_combinations):
        # Create compact abbreviations for naming
        dist_abbr = {'boolean_cube': 'BC'}[dist_type]
        
        # Create a compact name for this combination
        if is_scaled:
            run_name = f"MSP_{dist_abbr}_d{d}_deg{max_degree}_{format_leap(leap)}_scale{scaling_factor}"
        else:
            run_name = f"MSP_{dist_abbr}_d{d}_deg{max_degree}_{format_leap(leap)}_fixed"
            
        print(f"\nProcess {rank} starting combination {idx+1}/{len(my_combinations)}: {run_name} (Exp {exp_num})")
        print(f"Experiment {exp_num}/{num_experiments} - generating different random function with same properties")
        
        # Set seed based on experiment number and rank for uniqueness
        seed = 42 + exp_num + rank * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            # Initialize GPU to ensure it's ready
            torch.cuda.empty_cache()
            
            # Generate data - use more efficient batching for high dimensions
            print(f"Process {rank} generating data on GPU {assigned_gpu}")
            
            # Set batch size based on dimension
            if d <= 32:
                gen_batch_size = 500000
            elif d <= 64:
                gen_batch_size = 300000
            elif d <= 128:
                gen_batch_size = 150000
            else:
                gen_batch_size = 75000
                
            X = generate_data(dist_type, train_size, d, device, batch_size=gen_batch_size)
            
            print(f"Process {rank}: Data generated with shape {X.shape}")
            
            # Create target function using improved algorithm
            print(f"Process {rank} creating target function on GPU {assigned_gpu}")
            target_function, msp_structure = create_target_function(d, leap, max_degree, device)
            
            # Process data - use efficient batching for high dimensions
            print(f"Process {rank} computing target function with {X.shape[0]} samples")
            
            # Determine optimal batch size based on dimension and leap
            if d <= 32:
                batch_size = 100000
            elif d <= 64:
                batch_size = 75000
            elif d <= 128:
                batch_size = 50000
            else:
                batch_size = 25000
            
            # Adjust batch size based on max_degree and leap
            complexity_factor = max_degree * leap
            if complexity_factor > 50:
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
                    y_batch = generate_batch_efficiently(target_function, X[start_idx_batch:end_idx_batch], device)
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
                        sub_y = generate_batch_efficiently(target_function, X[j:sub_end], device)
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
            
            # Rest of the saving logic
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if is_scaled:
                smart_name = f"MSP_{dist_abbr}_d{d}_deg{max_degree}_{format_leap(leap)}_scale{scaling_factor}_{exp_num}"
            else:
                smart_name = f"MSP_{dist_abbr}_d{d}_deg{max_degree}_{format_leap(leap)}_fixed_{exp_num}"
            
            data_subdir = f"MSP_{smart_name}_{timestamp}"
            data_save_dir = os.path.join(data_base_dir, data_subdir)
            
            try:
                os.makedirs(data_save_dir, exist_ok=True)
                
                # Convert msp_structure (sets) to lists for JSON serialization
                msp_structure_lists = [list(int(m) for m in monomial) for monomial in msp_structure]
                
                # Save metadata and results
                results = {
                    'hyperparameters': {
                        'distribution_type': dist_type,
                        'input_dim': int(d),
                        'max_degree': int(max_degree),
                        'expected_leap': int(leap),
                        'actual_leap': int(target_function.get_leap()),
                        'actual_max_degree': int(target_function.get_max_degree()),
                        'use_degree_scaling': bool(is_scaled),
                        'scaling_factor': float(scaling_factor) if is_scaled else None,
                        'fixed_max_degree': int(fixed_max_degree) if not is_scaled else None,
                        'train_size': int(train_size),
                        'experiment_number': int(exp_num),
                        'experiment_description': f"Random function #{exp_num} with same leap/degree properties",
                        'random_seed': int(seed),
                        'mpi_rank': int(rank),
                        'mpi_size': int(size),
                        'gpu_device': int(assigned_gpu)
                    },
                    'target_function': {
                        'total_monomials': int(target_function.total_monomials),
                        'coefficients': target_function.coefficients.cpu().tolist() if target_function.total_monomials > 0 else [],
                        'msp_structure': msp_structure_lists,
                        'monomial_degree_counts': target_function.degree_counts
                    },
                    'leap_verification': {
                        'verified': bool(verify_msp_structure(msp_structure, leap)),
                        'expected_leap': int(leap),
                        'actual_leap': int(target_function.get_leap())
                    }
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
            del X, y, target_function
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