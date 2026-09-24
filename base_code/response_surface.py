"""Response surface terms: atomic building blocks for mu0(x) and mu1(x), the noise-free
potential outcomes. Each function returns a single array (one additive term), not a full
mu0/mu1 pair - combine them with plain addition in main.py, e.g.

    mu0_shape = lambda x: constant(x, value=5.0) + linear(x, slope=2.0)
"""
import numpy as np


def constant(x: np.ndarray, *, value: float) -> np.ndarray:
    """f(x) = value for every unit."""
    return np.full(shape=x.shape[0], fill_value=value)


def linear(x: np.ndarray, *, slope: float) -> np.ndarray:
    """f(x) = slope * x."""
    return slope * x[:, 0]


def arctan_squared(x: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """f(x) = scale * arctan(x)**2."""
    return scale * np.arctan(x[:, 0]) ** 2


def step(x: np.ndarray, *, jump: float = 1.0, threshold: float) -> np.ndarray:
    """f(x) = jump * 1(x > threshold)."""
    return jump * (x[:, 0] > threshold)


def exponential(x: np.ndarray, *, scale: float = 1.0, multiplier: float = 1.0) -> np.ndarray:
    """f(x) = scale * exp(multiplier * x)."""
    return scale * np.exp(multiplier * x[:, 0])
