import torch
from sftm_pt import SFTM

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Choose device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32  # Use float32 consistently
    print(f"Using device: {device}")

    # Create a small test matrix A (n x d) and query vector x (d)
    n, d = 5000, 500  # Small dimensions for testing
    A = torch.randn(n, d, device=device, dtype=dtype)  # Random matrix
    x = torch.randn(d, device=device, dtype=dtype)     # Random query vector
    
    # Initialize SFTM with default parameters
    sftm = SFTM(
        A,
        temperature=1.0,
        multiplicative_error=0.3,
        failure_probability=0.1,
        verbose=True
    )
    
    # Compare exact and approximate softmax
    k = 8  # Number of top elements to return
    
    print("\nComputing exact softmax...")
    exact_indices, exact_probs, exact_log_norm = sftm.softmax(x, k)
    
    print("\nComputing adaptive softmax...")
    approx_indices, approx_probs, approx_log_norm = sftm.adaptive_softmax(x, k)
    
    # Print results
    print("\nResults:")
    print(f"Top {k} indices (exact):   {exact_indices}")
    print(f"Top {k} indices (approx):  {approx_indices}")
    print(f"\nProbabilities (exact):   {exact_probs}")
    print(f"Probabilities (approx):  {approx_probs}")
    print(f"\nLog norm (exact): {exact_log_norm:.4f}")
    print(f"Log norm (approx): {approx_log_norm:.4f}")
    
    # Calculate relative errors
    prob_error = torch.abs((exact_probs - approx_probs) / exact_probs)
    log_norm_error = torch.abs((exact_log_norm - approx_log_norm) / exact_log_norm)
    
    print(f"\nRelative errors:")
    print(f"Probability relative errors: {prob_error}")
    print(f"Log norm relative error: {log_norm_error:.4f}")

if __name__ == "__main__":
    main()