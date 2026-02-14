import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Tuple


def rejection_sampling(
        target_density: Callable[[np.ndarray], float],
        proposal_density: Callable[[np.ndarray], float],
        proposal_rvs: Callable[[], np.ndarray],
        M: float = 1.0,
        n_samples: int = 10_000,
        diagnostics: bool = False
    ) -> Tuple[np.ndarray, float]:
    """Perform rejection sampling to generate samples from the target distribution
    using the proposal distribution.
    """

    k = len(proposal_rvs())
    samples = np.zeros((n_samples, k))

    accept, reject = 0, 0
    for i in range(n_samples):
        while True:
            # sample from the proposal distrbution
            proposal_sample = proposal_rvs()
            proposal_density_val = proposal_density(proposal_sample)
            target_density_val = target_density(proposal_sample)

            u = np.random.uniform(0, M * proposal_density_val)

            # print(proposal_density_val, proposal_sample)
            # print(u, target_density_val)

            # accept or reject the sample
            if u <= target_density_val:
                samples[i] = proposal_sample
                accept += 1
                break
            else:
                reject += 1
            
        if accept % (n_samples // 10) == 0 and accept > 0:
            print(f"{accept} / {n_samples} samples accepted.")

    acceptance_rate = accept / (accept + reject)

    if diagnostics:
        _generate_diagnostics(
            samples,
            acceptance_rate,
            target_density,
            proposal_density,
            M,
        )

    return samples, acceptance_rate

    
def _generate_diagnostics(
        samples: np.ndarray,
        acceptance_rate: float,
        target_density: Callable[[float], float],
        proposal_density: Callable[[float], float],
        M: float,
        tol = 1e-5,
    ) -> None:
    """Plot the diagnostic plot for rejection sampling.
    """

    print(f"Acceptance Rate: {acceptance_rate:.4f}")

   # For each dimension, plot the histogram of the samples and the target density
    k = samples.shape[1]

    for j in range(k):
        plt.figure(figsize=(8, 6))
        x_min = np.min(samples[:, j]) - tol
        x_max = np.max(samples[:, j]) + tol
        xx = np.linspace(x_min, x_max, 100)
        yy_target = target_density(xx)
        yy_proposal = proposal_density(xx)

        # Normalize the densities for plotting
        dx = xx[1] - xx[0]
        z = np.sum(yy_target * dx)
        yy_target /= z
        yy_proposal /= z

        plt.plot(xx, yy_target, color='red', label='Target Density')
        plt.plot(xx, M * yy_proposal, color='blue', label='Scaled Proposal Density')
        plt.hist(samples[:, j], bins=30, density=True, alpha=0.6, color='black')

        plt.xlabel(f'X{j+1}')
        plt.ylabel('Density')
        plt.title(f'Diagnostic Plot for Dimension {j+1}')
        plt.legend()
        plt.grid()
        plt.show()

    assert np.all(yy_target <= M * yy_proposal), "M is too small, target density exceeds scaled proposal density."

def _diagnostic_plot_2d(
        samples: np.ndarray,
        acceptance_rate: float,
        target_density: Callable[[np.ndarray], float],
        proposal_density: Callable[[np.ndarray], float],
        M: float,
        tol = 1e-5,
    ) -> None:
    """Plot the diagnostic plot for rejection sampling in 2D.
    """
    assert samples.shape[1] == 2, "This function is only for 2D samples."

    print(f"Acceptance Rate: {acceptance_rate:.4f}")

    x_min = np.min(samples[:, 0]) - tol
    x_max = np.max(samples[:, 0]) + tol
    y_min = np.min(samples[:, 1]) - tol
    y_max = np.max(samples[:, 1]) + tol
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    target_density_vals = target_density(np.stack([xx, yy], axis=-1))
    proposal_density_vals = proposal_density(np.stack([xx, yy], axis=-1))
    # Normalize the densities for plotting
    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]
    z = np.sum(target_density_vals * dx * dy)

    target_density_vals /= z
    proposal_density_vals /= z

    # plt.figure(figsize=(8, 6))
    # plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, color='black', s=10)
    # plt.contourf(xx, yy, target_density_vals, levels=50, cmap='Reds')
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.title('Diagnostic Plot for 2D Rejection Sampling')
    # plt.grid()

    plt.figure(figsize=(8, 6))

    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, target_density_vals, cmap='Reds', alpha=0.6)
    ax.plot_surface(xx, yy, M * proposal_density_vals, cmap='Blues', alpha=0.6)
    ax.scatter(samples[:, 0], samples[:, 1], np.zeros_like(samples[:, 0]), alpha=0.6, color='black', s=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Density')
    ax.set_title('3D Diagnostic Plot for 2D Rejection Sampling')
    plt.show()
    
    assert np.all(target_density_vals <= M * proposal_density_vals), "M is too small, target density exceeds scaled proposal density."

def _test_diagnostic_plot_2d():
    from scipy.stats import multivariate_normal

    target_density = lambda x: 0.5 * multivariate_normal.pdf(x, mean=[-2, -2], cov=[[1, 0], [0, 1]]) + \
            0.5 * multivariate_normal.pdf(x, mean=[2, 2], cov=[[1, 0], [0, 1]])
    proposal_density = lambda x: multivariate_normal.pdf(x, mean=[0, 0], cov=[[5, 0], [0, 5]])
    proposal_rvs = lambda: np.random.multivariate_normal(mean=[0, 0], cov=[[5, 0], [0, 5]])
    M_init = 10.0

    samples, acceptance_rate = rejection_sampling(
        target_density=target_density,
        proposal_density=proposal_density,
        proposal_rvs=proposal_rvs,
        M=M_init,
        n_samples=10_000
    )

    _diagnostic_plot_2d(samples, acceptance_rate, target_density, proposal_density, M_init)

def _test_1():
    # sample from a normal mixture using a cauchy distribution as the proposal
    from scipy.stats import norm, cauchy

    mixing_ratio = 0.3
    target_density = lambda x: mixing_ratio * norm.pdf(x, loc=-2, scale=1) + \
            (1 - mixing_ratio) * norm.pdf(x, loc=2, scale=1)
    proposal = cauchy(loc=0, scale=3)
    M_init = 4.0

    samples, acceptance_rate = rejection_sampling(
        target_density=target_density,
        proposal_density=proposal.pdf,
        proposal_rvs=lambda : proposal.rvs(size=(1, 1)),
        M=M_init,
        n_samples=10_000,
        diagnostics=True
    )

def _test_2():
    # Sample from a distribution proportion to x^2 on [0, 1] using a uniform distribution as the proposal
    target_density = lambda x: x**2
    proposal_density = lambda x: np.ones_like(x)
    proposal_rvs = lambda: np.random.uniform(0, 1, size=(1, 1))
    M_init = 1.2

    samples, acceptance_rate = rejection_sampling(
        target_density=target_density,
        proposal_density=proposal_density,
        proposal_rvs=proposal_rvs,
        M=M_init,
        n_samples=10_000,
        diagnostics=True
    )

    
if __name__ == "__main__":
    # _test_1()
    # _test_2()

    _test_diagnostic_plot_2d()