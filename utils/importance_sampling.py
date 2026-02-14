import numpy as np

from typing import Callable

def importance_sampling(
        f: Callable[[np.ndarray], float],
        targets,
        proposals,
        n_samples: int = 10_000
    ) -> np.ndarray:
    """Perform importance sampling to generate importance samples from the target
    distributions using the proposal distributions.

    Assumes that targets and proposals are from scipy.stats
    """
    k = len(targets)
    samples = np.zeros((n_samples, k))
    ratios = np.zeros((n_samples, k))

    for i in range(k):
        samples[:, i] = proposals[i].rvs(size=n_samples)
        ratios[:, i] = targets[i].pdf(samples[:, i]) / proposals[i].pdf(samples[:, i])
    
    # Compute the weighted average of f(X) using the importance weights
    weights = np.prod(ratios, axis=1)
    weighted_samples = f(samples) * weights

    return weighted_samples