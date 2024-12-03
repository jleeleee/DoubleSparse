import os
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Any

from .constants import (
    DEBUG,
    DEV_BY,
    DEV_RATIO,
    NUM_BINS,
    IMPORTANCE,
    SIGMA_BUFFER,
)

def fpc(it: torch.Tensor, d: int) -> torch.Tensor:
    return torch.ceil(it / (1 + (it - 1) / d)).to(torch.int64)

def approx_sigma(
    A: torch.Tensor,
    x: torch.Tensor,
    num_samples: Any = None,
    importance: bool = IMPORTANCE,
    log_file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug", "log.txt"
    ),
    debug_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to approximate sigma.
    Currently, We return the "median" of the std for the arm pulls across all arms.

    :param A: Matrix A in the original paper
    :param x: Vector x in the original paper
    :param num_samples: number of samples to use for approximating sigma

    :returns: the sigma approximation and distribution
    """
    n, d = A.shape
    dist = torch.ones(d, device=A.device) / d  # initially uniform

    # default, get true sigma
    if num_samples is None:
        num_samples = d

    if importance:
        factor = 0.1
        importance = torch.abs(x) / torch.sum(torch.abs(x))
        dist = (1 - factor) * importance + factor * dist
        dist = dist / dist.sum()  # for numerical stability

    # Generate random indices according to distribution
    coordinates = torch.multinomial(dist, num_samples, replacement=False)
    elmul = A[:, coordinates] * x[coordinates] / dist[coordinates]
    
    # Calculate standard deviation across dimension 1 and take mean
    sigma = torch.mean(torch.std(elmul, dim=1))

    # for toy example case where sigma = 0
    sigma = torch.max(sigma, torch.tensor(SIGMA_BUFFER, device=sigma.device))

    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write(f"sigma: {sigma.item()}\n")

        # get fraction of deviations that devitate by DEV_BY std (per arms)
        mus = torch.mean(elmul, dim=1).reshape(-1, 1)
        devs = torch.abs(elmul - mus) / sigma.reshape(-1, 1)
        num_devs = torch.sum(devs > DEV_BY, dim=0)
        fraction_per_arms = num_devs / n

        # Convert to numpy for matplotlib plotting
        fraction_per_arms_np = fraction_per_arms.cpu().numpy()
        
        # plot histogram
        bin_edges = torch.linspace(0.0, 1.0, NUM_BINS + 1).numpy()
        _, bins, _ = plt.hist(fraction_per_arms_np, bins=bin_edges, edgecolor="black")
        threshold_x = bins[int(DEV_RATIO * NUM_BINS)]
        num_outliers = torch.nonzero(fraction_per_arms > DEV_RATIO)[0]

        plt.axvline(x=threshold_x, color="red", linestyle="dashed")
        plt.xlabel("fraction")
        plt.ylabel("column frequency")
        plt.title(f"columns with fraction of arms greater than {DEV_BY} std")
        plt.text(0.95, 0.95, f"ratio of outliers: {len(num_outliers)/n:.2f}")
        plt.savefig(os.path.join(debug_path, "variance_of_columns.png"))
        plt.close()

    return sigma, dist

def get_importance_errors(
    mu: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    log_file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug", "log.txt"
    ),
) -> Tuple[float, float]:
    norm_mu = mu - mu.max()

    true_alpha = torch.exp(beta * norm_mu)
    true_alpha = true_alpha / torch.sum(true_alpha)
    alpha = alpha / torch.sum(alpha)
    alpha_error = torch.mean(alpha / true_alpha).item()

    true_gamma = torch.exp((beta * norm_mu) / 2)
    true_gamma = true_gamma / torch.sum(true_gamma)
    gamma = gamma / torch.sum(gamma)
    gamma_error = torch.mean(gamma / true_gamma).item()

    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write("(alpha, gamma error): ")
            f.write(f"({alpha_error}, {gamma_error})")
            f.write("\n")

    return alpha_error, gamma_error

def get_fs_errors(
    mu: torch.Tensor,
    mu_hat: torch.Tensor,
    beta: float,
    log_file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug", "log.txt"
    ),
) -> Tuple[float, float]:
    f_error = torch.sum(torch.exp(beta * mu_hat) * (beta * (mu - mu_hat)))
    f_error = (f_error / torch.sum(torch.exp(mu))).item()

    s_error = torch.sum(torch.exp(mu_hat) * (beta**2 * (mu - mu_hat) ** 2))
    s_error = (s_error / torch.sum(torch.exp(mu))).item()
    
    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write(f"(first order, second order): {f_error, s_error}\n")

    return f_error, s_error

def plot_norm_budgets(
    d: float,
    budget: torch.Tensor,
    a_error: torch.Tensor,
    g_error: torch.Tensor,
    f_error: torch.Tensor,
    s_error: torch.Tensor,
    debug_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug"),
):
    text = f"alpha err: {a_error:.3f}\n"
    text += f"gamma err: {g_error:.3f}\n"
    text += f"fo err: {f_error:.3f}\n"
    text += f"so error: {s_error:.3f}\n"

    # Convert to numpy for plotting
    budget_np = budget.cpu().numpy()
    
    bin_edges = torch.linspace(0.0, 1.0, NUM_BINS + 1).numpy()
    plt.hist(budget_np / d, bins=bin_edges, edgecolor="black")

    plt.xlabel("ratio of d")
    plt.ylabel("number of arms")
    plt.title("arm pulls for adaptive sampling")
    plt.text(0.95, 0.95, text)

    plt.savefig(os.path.join(debug_path, "normalization_budget.png"))
    plt.close()

def compare_true_arms(
    mu: torch.Tensor,
    best_arms: torch.Tensor,
    log_file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug", "log.txt"
    ),
) -> Tuple[torch.Tensor, torch.Tensor]:
    best_arms, _ = torch.sort(best_arms)
    true_best_arms = torch.argsort(mu)[-len(best_arms):]
    true_best_arms, _ = torch.sort(true_best_arms)
    diffs = mu[best_arms] - mu[true_best_arms]

    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write(f"algo arms <-> true arms: {best_arms.tolist()} <-> {true_best_arms.tolist()}\n")
            f.write(f"difference in mu for these arms: {diffs.tolist()}\n")

    return true_best_arms, diffs