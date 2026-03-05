import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

from .utility import generate_mean_and_std_err

def importance_sampling(
        target_density: Callable[[np.ndarray], float],
        proposal_density: Callable[[np.ndarray], float],
        proposal_rvs: Callable[[], np.ndarray],
        n_samples: int = 10_000,
        diagnostics: bool = False,
        truncate_q: float = 0.01,
    ) -> np.ndarray:
    """Perform importance sampling to generate samples from the target distribution
    using the proposal distribution.
    """
    samples = proposal_rvs(n_samples)
    ratios = target_density(samples) / proposal_density(samples)

    if diagnostics:
        _importance_sampling_diagnostics(
            samples,
            ratios,
            target_density,
            proposal_density,
            truncate_q,
        )

    return samples, ratios

def _importance_sampling_diagnostics(
        samples: np.ndarray,
        weights: np.ndarray,
        target_density: Callable[[np.ndarray], float],
        proposal_density: Callable[[np.ndarray], float],
        truncate_q: float = 0.01,
        tol: float = 0.1,
    ) -> None:
    """Generate diagnostics for importance sampling, including plots of the
    target and proposal densities, and the distribution of weights.
    """
    # Plot target and proposal densities

    ## compute the truncation quantiles for the samples
    lower_q = np.quantile(samples, truncate_q)
    upper_q = np.quantile(samples, 1 - truncate_q)

    truncation_mask = (samples >= lower_q) & (samples <= upper_q)
    samples = samples[truncation_mask]
    weights = weights[truncation_mask]

    x_min = min(np.min(samples), -tol)
    x_max = max(np.max(samples), tol)
    xx = np.linspace(x_min, x_max, 1000)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(xx, target_density(xx), label='Target Density', color='red')
    plt.plot(xx, proposal_density(xx), label='Proposal Density', color='blue')
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='gray', label='Samples')
    plt.title('Target vs Proposal Densities')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()

    # Plot distribution of weights
    plt.subplot(1, 2, 2)
    plt.hist(weights, bins=30, density=True, alpha=0.6, color='orange')
    plt.title('Distribution of Importance Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.grid()
    plt.show()


def self_normalised_importance_sampling(
        target_density: Callable[[np.ndarray], float], # Does not need to be normalised
        proposal_density: Callable[[np.ndarray], float], # Does not need to be normalised but needs to be sampleable
        proposal_rvs: Callable[[], np.ndarray],
        n_samples: int = 10_000,
        diagnostics: bool = False,
        truncate_q: float = 0.01,
    ) -> np.ndarray:
    """Perform self-normalised importance sampling to generate samples from the
    target distribution using the proposal distribution.
    """
    samples, ratios = importance_sampling(
        target_density,
        proposal_density,
        proposal_rvs,
        n_samples,
        diagnostics,
        truncate_q,
    )

    w = ratios / np.sum(ratios)

    if diagnostics:
        ess = 1 / np.sum(w**2)
        print(f"Effective Sample Size (ESS): {ess:.2f} out of {n_samples}")

    return samples, w

def _test_importance_sampling():
    from scipy.stats import norm, cauchy
    # Estimate the probability that X ~ N(0, 1) is greater than 4 
    samples, weights = importance_sampling(
        lambda x: norm.pdf(x, loc=0, scale=1),
        lambda x: cauchy.pdf(x, loc=4, scale=1),
        lambda n: cauchy.rvs(loc=4, scale=1, size=(n, 1)),
        n_samples=10_000,
        diagnostics=True,
        truncate_q=0.01,
    )

    weighted_samples = samples.flatten() * weights
    vals = weighted_samples > 4
    generate_mean_and_std_err(vals, prob=True)


def _test_self_normalised_importance_sampling():
    # Estimate the variance of studnent t distribution, with gaussian proposal
    from scipy.stats import norm
    target_density = lambda x: (1 + x**2 / 3)**(-2) # Unnormalised
    proposal_density = lambda x: norm.pdf(x)
    proposal_rvs = lambda n: norm.rvs(size=(n, 1))

    samples, w = self_normalised_importance_sampling(
        target_density,
        proposal_density,
        proposal_rvs,
        n_samples=100_000,
        diagnostics=True,
        truncate_q=0.01,
    )

    mean_est = np.sum(samples * w)
    var_est = np.sum((samples - mean_est)**2 * w)
    print(f"Estimated Variance: {var_est:.4f}")


    from scipy.stats import cauchy
    proposal_density = lambda x: cauchy.pdf(x)
    proposal_rvs = lambda n: cauchy.rvs(size=(n, 1))

    samples, w = self_normalised_importance_sampling(
        target_density,
        proposal_density,
        proposal_rvs,
        n_samples=100_000,
        diagnostics=True,
        truncate_q=0.01,
    )

    mean_est = np.sum(samples * w)
    vals = (samples - mean_est)**2 * w
    var_est = np.sum(vals)
    print(f"Estimated Variance: {var_est:.4f}")

    
if __name__ == "__main__":
    # _test_importance_sampling()
    _test_self_normalised_importance_sampling()