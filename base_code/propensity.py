"""Propensity functions: e(x) = P(A=1 | X=x), used to assign treatment on the training sample.
Each propensity score function takes one covariate coordinate (a 1-D array, shape (n,)) - pick
the coordinate in the running script, e.g.

    propensity_shape = lambda x: linear(1 - x[:, 0])
"""
import numpy as np


def _check_1d(x: np.ndarray) -> np.ndarray:
    """Fail loudly if a full covariate matrix is passed instead of a single coordinate x[:, j]."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected one covariate coordinate of shape (n,), e.g. x[:, 0]; got shape {x.shape}.")
    return x


def get_assignment(e: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Draw a ~ Bernoulli(e(x)) given propensity scores e. Shared across every propensity score function below."""
    return rng.binomial(n=1, p=e)


# --- Propensity score functions: e(x) = P(A=1 | X=x) ---


def constant(x: np.ndarray, *, p: float) -> np.ndarray:
    """e(x) = p for every unit, independent of x (e.g. p=0.5 for a randomized experiment)."""
    x = _check_1d(x)
    return np.full(shape=x.shape[0], fill_value=p)


def linear(x: np.ndarray) -> np.ndarray:
    """e(x) = x, clipped to [0.05, 0.95] so it is a valid probability even when x ranges outside it."""
    x = _check_1d(x)
    return np.clip(x, 0.05, 0.95)
