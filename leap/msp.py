import torch
import numpy as np
import random
import time
from typing import List, Tuple, Set, Dict, Optional
import itertools
import math
import scipy

def verify_leap_gpu(monomials: List[Tuple[int, ...]], device=None) -> int:
    """
    Compute the leap complexity of a set of monomials using GPU acceleration.
    
    Args:
        monomials: List of monomials represented as tuples of variable indices
        device: GPU device to use
        
    Returns:
        Leap complexity
    """
    if not monomials:
        return 0
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to sets for easier manipulation
    basis_sets = [set(m) for m in monomials]
    num_monomials = len(basis_sets)
    
    # For small functions, check all permutations directly
    if num_monomials <= 8:
        permutations = list(itertools.permutations(range(num_monomials)))
        min_max_leap = float('inf')
        
        for perm in permutations:
            current_support = set()
            max_leap = 0
            
            for i in perm:
                new_coords = basis_sets[i] - current_support
                max_leap = max(max_leap, len(new_coords))
                current_support.update(basis_sets[i])
                
            min_max_leap = min(min_max_leap, max_leap)
        
        return min_max_leap
    
    # For larger functions, create a matrix representation
    # Find all variables used in the monomials
    all_vars = set()
    for monomial in monomials:
        all_vars.update(monomial)
    all_vars = sorted(all_vars)
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    
    # Create a binary matrix where rows are monomials and columns are variables
    # matrix[i, j] = 1 if variable j is in monomial i
    matrix = torch.zeros((num_monomials, len(all_vars)), dtype=torch.int8, device=device)
    for i, monomial in enumerate(monomials):
        for var in monomial:
            matrix[i, var_to_idx[var]] = 1
    
    # Calculate leap matrix: leap_matrix[i, j] = leap from monomial i to j
    # This is the number of variables in j that are not in i
    # We use matrix operations to compute this in parallel
    leap_matrix = torch.zeros((num_monomials, num_monomials), dtype=torch.int, device=device)
    
    # For each pair of monomials i, j:
    # leap[i, j] = sum(matrix[j] & ~matrix[i])
    for i in range(num_monomials):
        not_in_i = 1 - matrix[i].unsqueeze(0)  # Shape: [1, num_vars]
        new_vars = matrix * not_in_i  # Shape: [num_monomials, num_vars]
        leap_matrix[i] = torch.sum(new_vars, dim=1)  # Shape: [num_monomials]
    
    # Find path with minimum maximum leap
    # We use a parallel Monte Carlo approach for large monomial sets
    num_trials = min(1000000, 100 * scipy.special.factorial(min(num_monomials, 10)))
    
    # Use GPU to generate and evaluate multiple random permutations in parallel
    min_max_leap = float('inf')
    batch_size = 10000  # Process permutations in batches
    
    for batch_start in range(0, num_trials, batch_size):
        batch_end = min(batch_start + batch_size, num_trials)
        current_batch_size = batch_end - batch_start
        
        # Generate random permutations
        perms = torch.zeros((current_batch_size, num_monomials), dtype=torch.long, device=device)
        for i in range(current_batch_size):
            perms[i] = torch.tensor(np.random.permutation(num_monomials), device=device)
        
        # Calculate max leap for each permutation
        max_leaps = torch.zeros(current_batch_size, dtype=torch.int, device=device)
        
        for perm_idx in range(current_batch_size):
            perm = perms[perm_idx]
            
            # Compute max leap for this permutation
            max_leap_perm = 0
            current_support = torch.zeros(len(all_vars), dtype=torch.bool, device=device)
            
            for i in range(num_monomials):
                monomial_idx = perm[i].item()
                monomial_vars = matrix[monomial_idx]
                
                # New variables not in current support
                new_vars = monomial_vars * (~current_support)
                new_var_count = torch.sum(new_vars).item()
                
                max_leap_perm = max(max_leap_perm, new_var_count)
                current_support = current_support | (monomial_vars.bool())
            
            max_leaps[perm_idx] = max_leap_perm
        
        # Update min_max_leap
        batch_min = torch.min(max_leaps).item()
        min_max_leap = min(min_max_leap, batch_min)
    
    return min_max_leap

def generate_candidate_monomials_gpu(variables: List[int], target_leap: int, 
                                    min_monomial_size: int, num_candidates: int,
                                    device=None) -> List[Tuple[int, ...]]:
    """
    Generate candidate monomials using GPU-accelerated matrix operations.
    
    Args:
        variables: List of variables to use
        target_leap: Target leap complexity
        min_monomial_size: Minimum size of largest monomial
        num_candidates: Number of candidate monomials to generate
        device: GPU device to use
        
    Returns:
        List of candidate monomials
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_vars = len(variables)
    
    # Create a random binary matrix where each column represents a monomial
    # matrix[i, j] = 1 if variable i is in monomial j
    matrix = torch.zeros((num_vars, num_candidates), dtype=torch.int8, device=device)
    
    # First column is the max monomial with min_monomial_size variables
    if min_monomial_size > 0:
        matrix[:min_monomial_size, 0] = 1
    
    # For leap 1 (staircase), create specific patterns
    if target_leap == 1:
        # Start with individual variables
        for i in range(1, min(num_vars + 1, num_candidates)):
            matrix[i-1, i] = 1
        
        # Then add staircase patterns (each monomial adds one more variable)
        idx = num_vars + 1
        for size in range(2, min(num_vars + 1, 10)):  # Up to size 10 to manage memory
            if idx >= num_candidates:
                break
            
            # For each size, generate multiple staircases
            for _ in range(min(100, num_candidates - idx)):
                # Choose 'size' random variables
                var_indices = torch.tensor(random.sample(range(num_vars), size), device=device)
                matrix[var_indices, idx] = 1
                idx += 1
    
    # For leap 2, generate pairs and triples
    elif target_leap == 2:
        idx = 1
        
        # Single variables
        for i in range(num_vars):
            if idx >= num_candidates:
                break
            matrix[i, idx] = 1
            idx += 1
        
        # Pairs of variables
        for i in range(num_vars):
            for j in range(i+1, num_vars):
                if idx >= num_candidates:
                    break
                matrix[i, idx] = 1
                matrix[j, idx] = 1
                idx += 1
        
        # Some triples and larger groups with leap <= 2
        while idx < num_candidates:
            # Generate a random subset of variables
            size = random.randint(3, min(num_vars, 10))
            var_indices = torch.tensor(random.sample(range(num_vars), size), device=device)
            matrix[var_indices, idx] = 1
            idx += 1
    
    # For leap > 2, generate more diverse patterns
    else:
        idx = 1
        
        # Generate subsets of different sizes
        for size in range(1, min(target_leap + 3, num_vars + 1)):
            # Number of subsets of this size to generate
            num_subsets = min(10000, num_candidates - idx)
            
            for _ in range(num_subsets):
                if idx >= num_candidates:
                    break
                
                # Choose 'size' random variables
                var_indices = torch.tensor(random.sample(range(num_vars), size), device=device)
                matrix[var_indices, idx] = 1
                idx += 1
    
    # Generate rest of candidates with random patterns
    while idx < num_candidates:
        # Choose a random size for the monomial
        size = random.randint(1, min(target_leap * 2, num_vars))
        var_indices = torch.tensor(random.sample(range(num_vars), size), device=device)
        matrix[var_indices, idx] = 1
        idx += 1
    
    # Convert matrix columns to monomials
    candidate_monomials = []
    for i in range(num_candidates):
        indices = torch.where(matrix[:, i] == 1)[0].cpu().numpy()
        if len(indices) > 0:
            monomial = tuple(sorted(variables[idx] for idx in indices))
            candidate_monomials.append(monomial)
    
    # Remove duplicates
    candidate_monomials = list(set(candidate_monomials))
    
    return candidate_monomials

def find_optimal_subset_gpu(candidate_monomials: List[Tuple[int, ...]], target_leap: int,
                          min_terms: int, max_terms: int, 
                          min_monomial_size: int, num_trials: int = 10000,
                          device=None) -> List[Tuple[int, ...]]:
    """
    Find a subset of monomials with the target leap complexity.
    
    Args:
        candidate_monomials: List of candidate monomials
        target_leap: Target leap complexity
        min_terms, max_terms: Range for number of monomials to select
        min_monomial_size: Minimum size of largest monomial
        num_trials: Number of trials
        device: GPU device to use
        
    Returns:
        List of monomials with the desired leap complexity
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure we have the max monomial
    max_monomial = None
    for monomial in candidate_monomials:
        if len(monomial) >= min_monomial_size:
            max_monomial = monomial
            break
    
    if max_monomial is None and min_monomial_size > 0:
        # If no monomial meets the size requirement, create one
        all_vars = set()
        for monomial in candidate_monomials:
            all_vars.update(monomial)
        
        if len(all_vars) >= min_monomial_size:
            max_monomial = tuple(sorted(list(all_vars)[:min_monomial_size]))
            candidate_monomials.append(max_monomial)
    
    # Create a function to evaluate a subset of monomials
    def evaluate_subset(subset):
        # Calculate leap
        leap = verify_leap_gpu(subset, device)
        
        # Calculate score
        unique_vars = len(set(v for m in subset for v in m))
        largest_monomial = max(len(m) for m in subset) if subset else 0
        # Inside multi_gpu_generate_leap_function, before the scoring calculation:
        min_monomial_size = max(d_max // 3, 1)  # Ensure min_monomial_size is defined



        
        score = (
            1000 * (leap <= target_leap) +  # Valid leap is most important
            100 * (largest_monomial >= min_monomial_size) +  # Meeting monomial size requirement
            unique_vars +  # More unique variables is better
            len(subset)  # More monomials is better (diversity)
        )
        
        return leap, score, subset
    
    # Try many random subsets
    best_leap = float('inf')
    best_score = -float('inf')
    best_subset = []
    
    for _ in range(num_trials):
        # Choose a number of monomials between min_terms and max_terms
        num_monomials = random.randint(min_terms, min(max_terms, len(candidate_monomials)))
        
        # Select random subset
        subset = random.sample(candidate_monomials, num_monomials)
        
        # Ensure max monomial is included if it exists
        if max_monomial and max_monomial not in subset:
            subset[0] = max_monomial
        
        # Evaluate
        leap, score, subset = evaluate_subset(subset)
        
        # Update best if better
        if score > best_score:
            best_score = score
            best_subset = subset
            best_leap = leap
        
        # If we found a valid solution, we can return early
        if leap <= target_leap and len(subset) >= min_terms:
            return subset
    
    return best_subset

def generate_leap_function_gpu(d: int, d_max: int, leap: int, 
                             min_terms: int = 3, max_terms: int = 10,
                             min_coverage: float = 0.5,
                             device=None) -> List[Tuple[int, ...]]:
    """
    Generate a random MSP function with the specified leap complexity using GPU.
    
    Args:
        d: Total number of variables (dimension)
        d_max: Maximum number of coordinates to use
        leap: Desired leap complexity
        min_terms, max_terms: Range for number of monomials
        min_coverage: Minimum fraction of d_max variables that must be used
        device: GPU device to use
        
    Returns:
        List of tuples representing monomials
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    min_monomial_size = max(d_max // 3, 1)
    print(f"Required monomial size: {min_monomial_size}")
    print(f"Required variables: {int(d_max * min_coverage)}")
    print(f"Target leap: {leap}")
    print(f"Using device: {device}")
    
    # Number of variables to use
    num_vars = max(int(d_max * min_coverage), min_monomial_size)
    
    # Select variables to use
    variables = list(range(1, d_max + 1))
    random.shuffle(variables)
    used_vars = variables[:num_vars]
    
    # Generate candidate monomials
    num_candidates = 100000  # Use a large number for better coverage
    candidate_monomials = generate_candidate_monomials_gpu(
        used_vars, leap, min_monomial_size, num_candidates, device
    )
    
    print(f"Generated {len(candidate_monomials)} candidate monomials")
    
    # Find optimal subset
    monomials = find_optimal_subset_gpu(
        candidate_monomials, leap, min_terms, max_terms, 
        min_monomial_size, num_trials=50000, device=device
    )
    
    # Verify the leap
    actual_leap = verify_leap_gpu(monomials, device)
    
    print(f"Generated MSP function with {len(monomials)} monomials")
    print(f"Actual leap complexity: {actual_leap}")
    print(f"Unique variables: {len(set(v for m in monomials for v in m))}")
    print(f"Monomial sizes: {[len(m) for m in monomials]}")
    
    return monomials

def multi_gpu_generate_leap_function(d: int, d_max: int, leap: int, 
                                   min_terms: int = 3, max_terms: int = 10,
                                   min_coverage: float = 0.5):
    """
    Generate a random MSP function using multiple GPUs.
    
    Args:
        d: Total number of variables (dimension)
        d_max: Maximum number of coordinates to use
        leap: Desired leap complexity
        min_terms, max_terms: Range for number of monomials
        min_coverage: Minimum fraction of d_max variables that must be used
        
    Returns:
        List of tuples representing monomials
    """
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        # If only one GPU, just use it directly
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return generate_leap_function_gpu(d, d_max, leap, min_terms, max_terms, min_coverage, device)
    
    print(f"Using {num_gpus} GPUs for parallel computation")
    
    # Launch parallel attempts on each GPU
    results = []
    
    # Simple multi-GPU approach: try multiple seeds on different GPUs
    for gpu_id in range(num_gpus):
        # Set a different seed for each GPU
        seed = int(time.time()) % 10000 + gpu_id * 1000
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Generate function on this GPU
        monomials = generate_leap_function_gpu(d, d_max, leap, min_terms, max_terms, min_coverage, device)
        actual_leap = verify_leap_gpu(monomials, device)
        
        # Calculate score
        unique_vars = len(set(v for m in monomials for v in m))
        largest_monomial = max(len(m) for m in monomials) if monomials else 0
        min_monomial_size = max(d_max // 3, 1)  # Ensure min_monomial_size is defined

        
        score = (
            1000 * (actual_leap <= leap) +  # Valid leap is most important
            100 * (largest_monomial >= min_monomial_size) +  # Meeting monomial size requirement
            unique_vars +  # More unique variables is better
            len(monomials)  # More monomials is better (diversity)
        )
        
        results.append((monomials, actual_leap, score))
    
    # Select the best result
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by score
    
    best_monomials, best_leap, best_score = results[0]
    
    print(f"Best result: leap={best_leap}, score={best_score}")
    
    return best_monomials

def format_msp_function(monomials: List[Tuple[int, ...]]) -> str:
    """Format the MSP function for display"""
    if not monomials:
        return "0"
        
    if len(monomials) > 10:
        first_terms = " + ".join([" * ".join(f"z{var}" for var in monomial) for monomial in monomials[:3]])
        last_terms = " + ".join([" * ".join(f"z{var}" for var in monomial) for monomial in monomials[-3:]])
        return f"{first_terms} + ... ({len(monomials)-6} more terms) ... + {last_terms}"
    else:
        return " + ".join([" * ".join(f"z{var}" for var in monomial) for monomial in monomials])

# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Test parameters
    d = 64       # Dimension
    d_max = 30    # Max variables to use
    leap = 8      # Leap complexity
    
    # Generate function using multiple GPUs
    start_time = time.time()
    monomials = multi_gpu_generate_leap_function(d, d_max, leap, min_coverage=1.0)
    generation_time = time.time() - start_time
    
    # Print results
    print(f"\nGenerated MSP function: {format_msp_function(monomials)}")
    print(f"Used {len(set(v for m in monomials for v in m))}/{d_max} variables")
    print(f"Monomial sizes: {[len(m) for m in monomials]}")
    print(f"Generation time: {generation_time:.4f} seconds")