import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

def generate_mean_and_std_err(
        data: np.ndarray,
        prob: bool = False
    ) -> None:
    """Calculate and print the mean and standard error of the given data.
    """
    n = len(data)
    mean = np.mean(data)

    if prob:
        std_err = np.sqrt(mean * (1 - mean) / n)
    else:
        std_err = np.std(data) / np.sqrt(n)

    print(f"Mean: {mean:.4f}")
    print(f"Standard Error: {std_err:.4f}")

def plot_density(
        data: np.ndarray,
        density: Callable[[float], float],
        x_min: float = None,
        x_max: float = None
    ) -> None:
    """Plot the density of the given data using a histogram and a density curve.
    """
    xx = np.linspace(x_min, x_max, 100)
    yy = density(xx)

    # Normalize the density curve to match the histogram
    dx = xx[1] - xx[0]
    z = np.sum(yy * dx)

    yy /= z

    plt.plot(xx, yy, color='red', label='Density Curve')
    plt.hist(data, bins=30, density=True, alpha=0.6, color='black')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Density Plot')

    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    # Example usage
    data = np.random.normal(loc=0, scale=1, size=10_000)
    density = lambda x: np.exp(-0.5 * x**2)

    generate_mean_and_std_err(data)
    plot_density(data, density=density, x_min=-4, x_max=4)