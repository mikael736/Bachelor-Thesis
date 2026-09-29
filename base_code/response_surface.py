"""Response surface terms: atomic building blocks for mu0(x) and mu1(x), the noise-free
potential outcomes. Each function takes one covariate coordinate (a 1-D array, shape (n,)) and
returns a single array (one additive term), not a full mu0/mu1 pair - pick the coordinate and
combine terms with plain arithmetic in the running script, e.g.

    mu0_shape = lambda x: constant(x[:, 0], value=5.0) + linear(x[:, 1], slope=2.0)
"""
import numpy as np


def _check_1d(x: np.ndarray) -> np.ndarray:
    """Fail loudly if a full covariate matrix is passed instead of a single coordinate x[:, j]."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected one covariate coordinate of shape (n,), e.g. x[:, 0]; got shape {x.shape}.")
    return x


def constant(x: np.ndarray, *, value: float) -> np.ndarray:
    """f(x) = value for every unit."""
    x = _check_1d(x)
    return np.full(shape=x.shape[0], fill_value=value)


def linear(x: np.ndarray, *, slope: float) -> np.ndarray:
    """f(x) = slope * x."""
    x = _check_1d(x)
    return slope * x


def arctan_squared(x: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """f(x) = scale * arctan(x)**2."""
    x = _check_1d(x)
    return scale * np.arctan(x) ** 2


def step(x: np.ndarray, *, jump: float = 1.0, threshold: float) -> np.ndarray:
    """f(x) = jump * 1(x > threshold)."""
    x = _check_1d(x)
    return jump * (x > threshold)


def exponential(x: np.ndarray, *, scale: float = 1.0, multiplier: float = 1.0) -> np.ndarray:
    """f(x) = scale * exp(multiplier * x)."""
    x = _check_1d(x)
    return scale * np.exp(multiplier * x)
