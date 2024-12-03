import torch
from hadamard_transform import hadamard_transform as ht
from math import ceil
from typing import Union, Tuple
from .constants import DEFAULT_VAR_PULL_INIT, DEFAULT_VAR_PULL_INCR

def generate_weighted_permutation(weights: torch.Tensor, gen=None):
    """
    Generate a weighed permutation using the Gumbel trick. Any size-k prefix of
    this permutation represents a weighted reservoir sample of size k.

    @param weights: The non-negative weights to use for the permutation
    @param gen: The random number generator seed
    @return: The permutation, the logits, and the perturbed logits
    """
    # print(weights)
    # assert torch.all(weights >= 0), 'Weights must be non-negative'
    
    if gen is not None:
        torch.manual_seed(gen)

    # Handle log(0) safely by using masked operations
    total_weight = weights.sum()
    # print(weights)
    # print(total_weight)
    mask = weights > 0
    logits = torch.full_like(weights, float('-inf'))
    logits[mask] = torch.log(weights[mask]) - torch.log(total_weight)
    
    # Generate Gumbel noise
    uniform = torch.rand_like(weights)
    gumbel_noise = -torch.log(-torch.log(uniform))
    perturbed_logits = logits + gumbel_noise
    
    # Get descending order permutation
    permutation = torch.argsort(perturbed_logits, descending=True)

    return permutation, logits, perturbed_logits

class BanditsSoftmax:
    """
    A class to handle the bandit problem used to perform adaptive softmax.

    This class performs pre-computation based on the provided atoms to reduce the 
    variance of the resulting arm pulls. Once a query is provided, the class can
    handle arm pulls and the resulting updates to the estimated mean of each 
    bandit arm for the adaptive softmax computation.

    Parameters
    ----------
    A : torch.Tensor
        The atom matrix A of shape (n, d) for the matrix-vector multiplication
    temperature : float, optional
        The temperature of the softmax (default 1.0)
    atom_importance_sampling : bool, optional
        Flag to enable atom-based importance sampling (default True)
    query_importance_sampling : bool, optional
        Flag to enable query-based importance sampling (default True)
    randomized_hadamard_transform : bool, optional
        Flag to enable randomized Hadamard transform (default False)
    verbose : bool, optional
        Flag to enable verbose output (default False)
    seed : int, optional
        The seed for random number generation (default 42)
    """
    def __init__(
        self,
        A: torch.Tensor,
        temperature: float = 1.0,
        noise_bound: float = None,
        atom_importance_sampling=False,
        query_importance_sampling=True,
        randomized_hadamard_transform=False,
        verbose=False,
        seed=42,
    ):
        assert len(A.shape) == 2, 'A must be a 2D tensor'
        
        self.device = A.device
        self.dtype = A.dtype
        self.n = A.shape[0]
        self.d = A.shape[1]
        self.temperature = temperature
        self.noise_bound = noise_bound
        self.atom_importance_sampling = atom_importance_sampling
        self.query_importance_sampling = query_importance_sampling
        self.randomized_hadamard_transform = randomized_hadamard_transform
        self.verbose = verbose

        self._A = A
        self._x = None
        
        torch.manual_seed(seed)

        if randomized_hadamard_transform:
            dp = 2 ** int(ceil(torch.log2(torch.tensor(self.d)).item()))
            pad_size = dp - self.d
            self._A = torch.nn.functional.pad(A, (0, pad_size), 'constant', 0)
            self.d = dp
            self._rademacher = torch.randint(2, (self.d,), device=self.device, dtype=self.dtype) * 2 - 1
            self._A = ht(self._A * self._rademacher)

        if atom_importance_sampling:
            self._atom_weights = torch.sum(torch.abs(self._A), dim=0)
        else:
            self._atom_weights = torch.ones(self.d, device=self.device)
            
        self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(self._atom_weights, gen=seed)
        
        # Calculate q and estimated atom sigma squared
        q = (self._atom_weights / self._atom_weights.sum()).unsqueeze(0)
        q = torch.where(q == 0, torch.ones_like(q), q)  # Handle zero weights
        self._est_atom_sig2 = torch.max(torch.sum((self._A / q / self.d) ** 2 * q, dim=1))
        self._est_query_sig2 = None
        self._sparse_columns = None

        self._Ap = None if self.query_importance_sampling else self._A[:, self._permutation].clone()
        self._xp = None

        # Initialize arrays with consistent dtype
        self._it = torch.zeros(self.n, dtype=torch.int, device=self.device)
        self._estimates = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        self._var = torch.full((self.n,), float('inf'), dtype=self.dtype, device=self.device)

        if self.verbose:
            print(f'BanditsSoftmax initialized with {self.n} arms and {self.d} dimensions')
            print(f'Atom importance sampling: {self.atom_importance_sampling}')
            print(f'Query importance sampling: {self.query_importance_sampling}')
            print(f'Randomized Hadamard transform: {self.randomized_hadamard_transform}')
            print(f'Permutation:\n{self._permutation}')

            if atom_importance_sampling:
                print(f'Atom weights:\n{self._atom_weights}')

            if randomized_hadamard_transform:
                print(f'Columns 0-padded: {A.shape[1]} --> {self.d}')

    @property
    def it(self):
        """
        The number of pulls for each arm.
        """
        return self._it
    
    @property
    def max_pulls(self):
        """
        The maximum number of times any arm can be pulled.
        """
        assert self._x is not None, 'Query vector not set'
        return self.d - self._num_sparse_columns
    
    @property
    def variance(self):
        """
        An upper bound of the variance of the bandit pulls.
        """
        assert self._x is not None, 'Query vector not set'
        
        if self.noise_bound is not None:
            return self.noise_bound
        
        return self._est_atom_sig2 * self._est_query_sig2 * (self.max_pulls ** 2) * (self.temperature ** 2)

    def set_query(self, x: torch.Tensor, seed=42):
        """
        Set the query vector for the bandit problem.
        
        @param x: The query vector
        @param seed: Random seed
        """
        if self.randomized_hadamard_transform:
            assert x.size(0) <= self.d, 'Query vector must be of size d or less if padding was performed'
        else:
            assert x.size(0) == self.d, f'Query vector must be of size {self.d} was x.size(0)={x.size(0)}'

        torch.manual_seed(seed)

        self._it = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self._estimates = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self._var = torch.full((self.n,), float('inf'), dtype=torch.float32, device=self.device)

        self._x = torch.nn.functional.pad(x, (0, self.d - x.size(0)), 'constant', 0)

        if self.randomized_hadamard_transform:
            self._x = ht(self._x * self._rademacher)

        if self.query_importance_sampling:
            query_weights = torch.abs(self._x)
            self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(
                query_weights * self._atom_weights, gen=seed
            )
        
        self._xp = self._x[self._permutation].clone()

        self._num_sparse_columns = torch.sum(self._logits == float('-inf')).item()
        n_nonzero = self.d - self._num_sparse_columns
        
        if self.query_importance_sampling:
            self._est_query_sig2 = torch.mean(torch.abs(self._xp[:n_nonzero])) ** 2
        else:
            self._est_query_sig2 = torch.mean(self._xp[:n_nonzero] ** 2)

        if self.verbose and self.query_importance_sampling:
            print(f'Query weights:\n{query_weights}')
            print(f'Combined weights:\n{self._atom_weights * query_weights}')
            print(f'Updated permutation:\n{self._permutation}')

    def exact_values(self, arms: torch.Tensor) -> torch.Tensor:
        """
        Compute the exact value for the specified arms using efficient PyTorch operations.
        
        @param arms: The arms for which to compute the exact value
        @return: The exact values of the specified arms
        """
        assert self._x is not None, 'Query vector not set'
        
        mask = self.it[arms] < self.max_pulls
        if torch.any(mask):
            if self._Ap is None:
                # Keep the same logical order as original code
                A_selected = self._A[arms]  # First get the arms
                # Then apply permutation to each selected arm's values
                A_arms = A_selected[:, self._permutation]
            else:
                A_arms = self._Ap[arms]
                
            # Keep original matrix multiplication
            result = (A_arms @ self._xp) * self.temperature
            self._estimates[arms] = result.to(dtype=self._estimates.dtype)
            self._it[arms] = self.max_pulls
            self._var[arms] = 0
        
        return self._estimates[arms]

    def pull_arm(self, arm: int, it: int) -> float:
        """
        Pull an arm the given number of times.
        
        @param arm: The arm to pull
        @param it: Number of times to pull the arm
        @return: The updated estimated value of the arm
        """
        assert self._x is not None, 'Query vector not set'
        return self.batch_pull(torch.tensor([arm], device=self.device), it)[0]

    def pull(self, arms: torch.Tensor, its: torch.Tensor) -> torch.Tensor:
        """
        Pull the specified arms the provided number of times.
        
        @param arms: The arms to pull
        @param its: Number of times to pull each arm
        @return: The updated estimated values of the specified arms
        """
        assert self._x is not None, 'Query vector not set'
        assert arms.size() == its.size(), 'The number of arms and pulls must be the same'

        to_pull = (its > self._it[arms]).nonzero().squeeze(-1)
        if to_pull.numel() > 0:
            for i in to_pull:
                self.batch_pull(arms[i].unsqueeze(0), its[i])
        
        return self._estimates[arms]

    def pull_to_var(
        self,
        arms: torch.Tensor,
        var_threshold: Union[float, torch.Tensor],
        init_pulls: int = DEFAULT_VAR_PULL_INIT,
        pull_mult: float = DEFAULT_VAR_PULL_INCR,
        fudge_factor_var: float = 1.0,
        max_pulls: int = float('inf'),
        batched: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pull arms until variance threshold is met.
        
        @param arms: The arms to pull
        @param var_threshold: Variance threshold
        @param init_pulls: Initial number of pulls
        @param pull_mult: Factor to increase pulls by
        @param fudge_factor_var: Variance fudge factor
        @param max_pulls: Maximum number of pulls allowed
        @param batched: Whether to use batch pulling
        @return: Updated estimates and variances
        """
        assert self._x is not None, 'Query vector not set'

        max_pulls = min(max_pulls, self.max_pulls)
        threshold_var = var_threshold / fudge_factor_var
        
        if isinstance(threshold_var, (float, int)):
            threshold_var = torch.tensor(threshold_var, device=self.device)
            
        to_pull = (self._var[arms] > threshold_var) & (self._it[arms] < max_pulls)
        num_pulls = min(init_pulls, max_pulls)

        while torch.any(to_pull):
            num_pulls_rounded = int(ceil(num_pulls))
            if batched:
                self.batch_pull(arms, num_pulls_rounded)
            else:
                pulling = arms[to_pull]
                self.pull(pulling, torch.full_like(pulling, num_pulls_rounded))
            to_pull = (self._var[arms] > threshold_var) & (self._it[arms] < max_pulls)
            num_pulls = min(max_pulls, num_pulls * pull_mult)

        return self._estimates[arms], self._var[arms] * fudge_factor_var

    def batch_pull(self, arms: torch.Tensor, it: int) -> torch.Tensor:
        """
        Pull specified arms in batch mode.
        
        @param arms: The arms to pull
        @param it: Number of times to pull each arm
        @return: The updated estimated values
        """
        assert self._x is not None, 'Query vector not set'
        assert torch.unique(self._it[arms]).numel() <= 1, 'All arms must have been pulled the same number of times'

        if self.verbose:
            print(f"Pulling arm(s):\n{arms}")
            print(f'Using {(it / self.max_pulls) * 100:.2f}% of the budget')

        if arms.numel() == 0 or it <= self._it[arms][0]:
            return self._estimates[arms]
        
        prev_it = self._it[arms][0].item()
        next_it = min(it, self.max_pulls)

        # importance sampling
        if self.atom_importance_sampling or self.query_importance_sampling:
            threshold = float('-inf') if next_it == self.max_pulls else self._perturbed_logits[self._permutation[next_it]]
            weights = 1 - torch.exp(-torch.exp(self._logits[self._permutation[:next_it]] - threshold))
            weights = torch.nan_to_num(weights, nan=1.0)

            if self._Ap is None:
                A = self._A[arms][:, self._permutation[:next_it]]
            else:
                A = self._Ap[arms, :next_it]
            
            A = A.reshape(len(arms), next_it)
            x = self._xp[:next_it] / weights
            
            # Convert result to match self._estimates dtype
            result = (A @ x) * self.temperature
            self._estimates[arms] = result.to(dtype=self._estimates.dtype)
            
            var_result = torch.sum((A * self._xp[:next_it] * self.temperature) ** 2 * (1 - weights) / (weights ** 2), dim=1)
            self._var[arms] = var_result.to(dtype=self._var.dtype)

        # no importance sampling (equal weighting)
        else:
            self._estimates[arms] *= prev_it
            result = (self._Ap[arms, prev_it:next_it] @ self._xp[prev_it:next_it]) * (self.max_pulls * self.temperature)
            self._estimates[arms] += result.to(dtype=self._estimates.dtype)
            self._estimates[arms] /= next_it
            self._var[arms] = (self.variance / next_it).to(dtype=self._var.dtype)

        if next_it == self.max_pulls:
            self._var[arms] = 0

        self._it[arms] = next_it

        return self._estimates[arms]