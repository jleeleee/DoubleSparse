import torch
from typing import Tuple, Optional, Union
from math import log, ceil, sqrt
from typing import Callable, Any

from .bandits_softmax_pt import BanditsSoftmax
from .utils_pt import fpc
from .constants import DEFAULT_CI_DECAY, TUNE_EXP_FUDGE_HIGH, TUNE_EXP_FUDGE_LOW, DEFAULT_CI_INIT
from torch import Tensor

def adaptive_softmax_batched(
    query: torch.Tensor,  # [batch_size, num_heads, seq_length_query, head_dim]
    key: torch.Tensor,    # [batch_size, num_heads, seq_length_key, head_dim ]
    top_k: Optional[int] = None,
) -> torch.Tensor:
    batch_size, num_heads, seq_length_query, head_dim = query.shape
    _, _, seq_length_key, _ = key.shape
    device = query.device
    dtype = query.dtype
    
    attn_weights = torch.zeros(batch_size, num_heads, seq_length_query, seq_length_key, device=device, dtype=dtype)
    for b in range(batch_size):
        for h in range(num_heads):
            curr_query = query[b, h]  # [seq_length_query, head_dim]
            curr_key = key[b, h]      # [seq_length_key, head_dim]
            sftm = SFTM(curr_key.to(dtype=torch.float32) , temperature=1.0)
            
            for i in range(seq_length_query):
                query_vec = curr_query[i].to(dtype=torch.float32)  # [head_dim]
                k = top_k if top_k is not None and curr_key.shape[0] > top_k else curr_key.shape[0]
                indices, probabilities, _ = sftm.adaptive_softmax(query_vec, k=k)
                attn_weights[b, h, i][indices] = probabilities.to(dtype=dtype)
                
    return attn_weights

def adaptiveSoftmax(
    input: Tensor,
    atom_matrix: Tensor,
    dim: Optional[int] = None,
    temperature: float = 1.0,
    multiplicative_error: float = 3e-1,
    failure_probability: float = 1e-1,
    dtype: Optional[torch.dtype] = None,
    k: Optional[int] = None
) -> Tensor:
    """Apply an adaptive softmax function.

    Adaptive softmax approximates the softmax function by examining only a subset of
    the input values, with provable PAC guarantees. It is particularly useful for
    large-scale matrix-vector multiplications followed by softmax.

    Args:
        input (Tensor): Input tensor to compute softmax over
        atom_matrix (Tensor): Matrix A for the matrix-vector multiplication
        dim (int, optional): Dimension along which adaptive softmax will be computed
        temperature (float, optional): Temperature parameter for scaling logits. Default: 1.0
        multiplicative_error (float, optional): Error tolerance epsilon. Default: 0.3
        failure_probability (float, optional): Failure probability delta. Default: 0.1
        dtype (torch.dtype, optional): Desired data type of returned tensor.
            If specified, the input tensor is cast to dtype before the operation
            is performed. Default: None
        k (int, optional): Number of top elements to return. If None, returns full distribution.
            Default: None

    Returns:
        Tensor: Tensor of the same shape as input with adaptive softmax probabilities

    Example::
        >>> atom_matrix = torch.randn(1000, 512)  # 1000 atoms of dimension 512
        >>> query = torch.randn(512)              # Query vector of dimension 512
        >>> probs = adaptive_softmax(query, atom_matrix, dim=0)
    """
    # Handle dtype conversion if specified
    if dtype is not None:
        input = input.to(dtype=dtype)
        atom_matrix = atom_matrix.to(dtype=dtype)

    # Default to last dimension if not specified
    if dim is None:
        dim = -1

    # Ensure input dimensions match atom_matrix
    if input.size(dim) != atom_matrix.shape[1]:
        raise ValueError(
            f"Input dimension {input.size(dim)} doesn't match atom matrix dimension {atom_matrix.shape[1]}"
        )

    # Initialize SFTM
    sftm = SFTM(
        atom_matrix,
        temperature=temperature,
        multiplicative_error=multiplicative_error,
        failure_probability=failure_probability
    )

    # If k is not specified, use full dimension
    if k is None:
        k = atom_matrix.shape[0]

    # Compute adaptive softmax
    indices, probabilities, _ = sftm.adaptive_softmax(input, k)

    # Create output tensor with zeros
    output = torch.zeros(atom_matrix.shape[0], dtype=input.dtype, device=input.device)
    
    # Place probabilities at correct indices
    output[indices] = probabilities

    return output

# Optional: provide a class-based interface similar to nn.Softmax
class AdaptiveSoftmax(torch.nn.Module):
    """Applies adaptive softmax to the input tensor.

    Args:
        atom_matrix (Tensor): Matrix A for the matrix-vector multiplication
        dim (int, optional): Dimension along which adaptive softmax will be computed
        temperature (float, optional): Temperature parameter. Default: 1.0
        multiplicative_error (float, optional): Error tolerance epsilon. Default: 0.3
        failure_probability (float, optional): Failure probability delta. Default: 0.1
        k (int, optional): Number of top elements to return. If None, returns full distribution.
            Default: None
    """
    def __init__(
        self,
        atom_matrix: Tensor,
        dim: Optional[int] = None,
        temperature: float = 1.0,
        multiplicative_error: float = 3e-1,
        failure_probability: float = 1e-1,
        k: Optional[int] = None
    ):
        super(AdaptiveSoftmax, self).__init__()
        self.atom_matrix = atom_matrix
        self.dim = dim
        self.temperature = temperature
        self.multiplicative_error = multiplicative_error
        self.failure_probability = failure_probability
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_softmax(
            input,
            self.atom_matrix,
            self.dim,
            self.temperature,
            self.multiplicative_error,
            self.failure_probability,
            k=self.k
        )

    def extra_repr(self) -> str:
        return f'dim={self.dim}, temperature={self.temperature}'
    
def logsumexp(tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
    """PyTorch implementation of logsumexp that handles -inf values safely"""
    if dim is None:
        max_val = tensor.max()
        return max_val + torch.log(torch.exp(tensor - max_val).sum())
    else:
        max_val = tensor.max(dim=dim, keepdim=True)[0]
        return max_val.squeeze(dim) + torch.log(torch.exp(tensor - max_val).sum(dim=dim))

def softmax(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch implementation of softmax that handles -inf values safely"""
    exp_tensor = torch.exp(tensor - tensor.max(dim=dim, keepdim=True)[0])
    return exp_tensor / exp_tensor.sum(dim=dim, keepdim=True)

class SFTM:
    """
    Softmax Fast Top-k via Monte Carlo (SFTM) approximates the softmax function
    following a matrix-vector multiplication looking at only a subset of the
    columns of the matrix and with provable PAC guarantees.
    """

    def __init__(
        self,
        A: torch.Tensor,
        temperature: float = 1.0,
        multiplicative_error: float = 3e-1,
        failure_probability: float = 1e-1,
        noise_bound: float = None,
        atom_importance_sampling: bool = True,
        query_importance_sampling: bool = True,
        randomized_hadamard_transform: bool = False,
        exact_pull_best_arm: bool = True,
        max_init_pull_budget: float = 1.0,
        verbose: bool = False,
        seed=42
    ):
        self.A = A
        self.device = A.device
        self.n = A.shape[0]
        self.d = A.shape[1]
        self.temperature = temperature
        self.multiplicative_error = multiplicative_error
        self.failure_probability = failure_probability
        self.exact_pull_best_arm = exact_pull_best_arm
        self.max_init_pull_budget = max_init_pull_budget
        self.verbose = verbose
        self.seed = seed

        if self.verbose:
            print(f"Initializing SFTM for a matrix of shape ({self.n} x {self.d})...")
            print("Parameters:")
            print(f"\t- temperature: {self.temperature}")
            print(f"\t- multiplicative_error: {self.multiplicative_error}")
            print(f"\t- failure_probability: {self.failure_probability}")

        self.bandits = BanditsSoftmax(
            A,
            temperature=temperature,
            noise_bound=noise_bound,
            atom_importance_sampling=atom_importance_sampling,
            query_importance_sampling=query_importance_sampling,
            randomized_hadamard_transform=randomized_hadamard_transform,
            verbose=verbose,
            seed=self.seed,
        )

        if self.verbose:
            print("SFTM initialized.")
            print("")

    def tune_fudge_factors(self, X_train: torch.Tensor, k: int = 1, verbose: bool = False) -> Tuple[float, float]:
        """
        Fits the fudge factors of SFTM based on the provided queries.
        """
        if verbose:
            print(f"Fitting SFTM fudge factors for {X_train.shape[0]} query vectors...")

        # get true best arms and log norm for each query
        MU = (X_train @ self.A.T) * self.temperature
        topk_values, topk_indices = torch.topk(MU, k, dim=1)
        TOP_K = torch.sort(topk_indices, dim=1)[0]
        LOG_NORM = logsumexp(MU, dim=1)

        delta = self.failure_probability
        eps = self.multiplicative_error

        # binary search for fudge factors
        def bin_search(f_check: Callable[[float, float, torch.Tensor, torch.Tensor, float], bool],
                      fudge_bandits=None, fudge_log_norm=None) -> float:
            target_success_rate = 1 - delta
            lo = TUNE_EXP_FUDGE_LOW
            hi = TUNE_EXP_FUDGE_HIGH
            
            while lo + 1e-2 < hi:
                mi = (lo + hi) / 2
                fudge_factor = 10 ** mi

                if verbose:
                    print(f"\tTrying fudge factor: {fudge_factor}")

                fails_remaining = int(delta * X_train.shape[0])
                for i, x in enumerate(X_train):
                    fudge_b = fudge_factor if fudge_bandits is None else fudge_bandits
                    fudge_ln = fudge_factor if fudge_log_norm is None else fudge_log_norm
                    fails_remaining -= not f_check(fudge_b, fudge_ln, x, TOP_K[i], LOG_NORM[i])
                    if fails_remaining < 0:
                        break

                if fails_remaining < 0:
                    lo = mi
                else:
                    hi = mi

            return 10 ** hi

        def f_check_bandits(fudge_bandits: float, _fudge_log_norm: float,
                          x: torch.Tensor, best_arms: torch.Tensor, _log_norm: float) -> bool:
            self.bandits.set_query(x)
            best_arms_hat = self.best_arms(delta / 2, k, fudge_factor=fudge_bandits)
            return torch.all(best_arms_hat == best_arms)

        def f_check_log_norm(fudge_bandits: float, fudge_log_norm: float,
                           x: torch.Tensor, best_arms: torch.Tensor, log_norm: float) -> bool:
            best_arms_hat, p_hat, _ = self.adaptive_softmax(
                x, 1, fudge_bandits=fudge_bandits, fudge_log_norm=fudge_log_norm
            )

            if not torch.all(best_arms_hat == best_arms):
                return False

            p = torch.exp((self.A @ x)[best_arms] - log_norm)
            err = torch.max(torch.abs((p_hat - p) / p))

            return err <= eps

        if verbose:
            print("Fitting bandits fudge factor...")

        fudge_bandits = bin_search(f_check_bandits)

        if verbose:
            print("Fitting log norm fudge factor...")

        fudge_log_norm = bin_search(f_check_log_norm, fudge_bandits=fudge_bandits)

        if self.verbose:
            print("Fitting complete.")
            print("")

        return fudge_bandits, fudge_log_norm

    def softmax(self, x: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the true softmax, returning the top-k indices and the softmax.
        """
        mu = (self.A @ x) * self.temperature
        topk_values, topk_indices = torch.topk(mu, k)
        top_k = torch.sort(topk_indices)[0]
        return top_k, softmax(mu)[top_k], logsumexp(mu)

    def adaptive_softmax(
        self,
        x: torch.Tensor,
        k: int = 1,
        fudge_bandits: float = 1.0,
        fudge_log_norm: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the approximate softmax using the SFTM algorithm.
        """
        if self.verbose:
            print(f"Computing adaptive softmax for query vector {x}...")
        self.bandits.set_query(x, seed=self.seed)

        eps = self.multiplicative_error
        delta = self.failure_probability

        delta_sub = delta / 2 if self.exact_pull_best_arm else delta / 3
        eps_sub = eps if self.exact_pull_best_arm else eps / 4

        # batched warmup
        V0 = 1 / (17 * log(6 * self.n / delta_sub))
        self.bandits.pull_to_var(
            torch.arange(self.n, device=self.device),
            V0,
            fudge_factor_var=fudge_log_norm,
            max_pulls=int(self.max_init_pull_budget * self.bandits.max_pulls),
            batched=True
        )

        i_star_hat = self.best_arms(delta_sub, k, fudge_factor=fudge_bandits)

        if self.exact_pull_best_arm:
            mu_star_hat = self.bandits.exact_values(i_star_hat)
        else:
            mu_star_hat = self.estimate_arm_logits(i_star_hat, eps_sub, delta_sub)

        log_S_hat = self.log_norm_estimation(eps, delta_sub, fudge_factor=fudge_log_norm)

        if self.verbose:
            print(f"Top-{k} arms: {i_star_hat}")
            print(f"Estimated logit values: {mu_star_hat}")
            print(f"Estimated log normalizing constant: {log_S_hat}")

        return i_star_hat, torch.exp(mu_star_hat - log_S_hat), log_S_hat

    def best_arms(
        self,
        delta: float,
        k: int,
        ci_decay: float = DEFAULT_CI_DECAY,
        fudge_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Finds the top-k arms with the highest estimated logit values.
        """
        if self.verbose:
            print(f"Finding top-{k} arms with the highest estimated logit values...")

        n = self.n
        d = self.bandits.max_pulls
        v = DEFAULT_CI_INIT * self.bandits.variance

        # initialize parameters
        confidence_set = torch.arange(n, device=self.device)

        while True:
            # pull arms and update confidence interval
            estimates, variances = self.bandits.pull_to_var(confidence_set, v, fudge_factor_var=fudge_factor, batched=True)
            confidence_intervals = torch.sqrt(2 * variances * log(6 * n * log(d) / delta))

            # update confidence set
            best_arm_hat = torch.argmax(estimates)
            keep = estimates + confidence_intervals >= estimates[best_arm_hat] - confidence_intervals[best_arm_hat]

            if self.verbose:
                print(f"Confidence intervals: {confidence_intervals}")
                print(f"Estimates: {estimates}")
                print(f"Confidence set: {confidence_set[keep]}")

            # check stopping condition
            if torch.sum(keep) <= k:
                break

            # update parameters
            confidence_set = confidence_set[keep]
            v *= ci_decay

        _, top_k_indices = torch.topk(estimates, k)
        return confidence_set[top_k_indices]

    def estimate_arm_logits(self, arms: torch.Tensor, eps: float, delta: float) -> torch.Tensor:
        """
        Estimates the logit values of the specified arms with PAC guarantees.
        """
        if self.verbose:
            print(f"Estimating logit values for arms {arms}...")
        V = 1 / (32 * log(2 / delta) / (eps ** 2))
        return self.bandits.pull_to_var(arms, V)[0]

    def log_norm_estimation(
        self,
        eps: float,
        delta: float,
        fudge_factor: float = 1.0,
        first_pull_batched: bool = False,
    ) -> torch.Tensor:
        """
        Estimates the log normalizing constant of the softmax function with PAC guarantees.
        """
        n = self.n

        V0 = 1 / (17 * log(6 * n / delta))

        if self.verbose:
            print("Estimating log normalizing constant of the softmax function...")
            print(f"Initial sample mean threshold: {V0}")

        # initial estimates
        mu_hat, var_hat = self.bandits.pull_to_var(
            torch.arange(n, device=self.device),
            V0,
            fudge_factor_var=fudge_factor,
            max_pulls=int(self.max_init_pull_budget * self.bandits.max_pulls),
            batched=first_pull_batched,
        )

        C = torch.sqrt(2 * log(6 * n / delta) * var_hat)

        if self.verbose:
            print(f"Initial estimates: {mu_hat}")

        log_alpha = (mu_hat - C)
        log_gamma = (mu_hat - C) / 2
        log_alpha_sum = logsumexp(log_alpha)
        log_gamma_sum = logsumexp(log_gamma)

        # adapt sample sizes based on initial estimates
        log_c = log(16 * sqrt(2) * log(6 * n / delta) / eps) + 2 * log_gamma_sum - log_alpha_sum
        log_d = log(16 * log(12 / delta) / (eps ** 2))

        V1 = torch.full((n,), float('inf'), device=self.device)
        V1 = torch.minimum(V1, torch.exp(log_gamma_sum - (log_c + log_gamma)))
        V1 = torch.minimum(V1, torch.exp(log_alpha_sum - (log_d + log_alpha)))

        if self.verbose:
            print(f"Adaptive variance ratio thresholds: {V1}")

        # make updated estimates
        mu_hat, _ = self.bandits.pull_to_var(
            torch.arange(n, device=self.device),
            V1,
            fudge_factor_var=fudge_factor
        )

        if self.verbose:
            print(f"Updated estimates: {mu_hat}")
            print(f"Estimated log normalizing constant: {logsumexp(mu_hat)}")

        return logsumexp(mu_hat)