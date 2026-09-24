"""Propensity functions: e(x) = P(A=1 | X=x), used to assign treatment on the training sample."""
import numpy as np


def get_assignment(e: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Draw a ~ Bernoulli(e(x)) given propensity scores e. Shared across every propensity score function below."""
    return rng.binomial(n=1, p=e)


# --- Propensity score functions: e(x) = P(A=1 | X=x) ---


def constant(x: np.ndarray, *, p: float) -> np.ndarray:
    """e(x) = p for every unit, independent of x (e.g. p=0.5 for a randomized experiment)."""
    return np.full(shape=x.shape[0], fill_value=p)


def linear(x: np.ndarray) -> np.ndarray:
    """e(x) = x, clipped to [0.05, 0.95] so it is a valid probability even when x ranges outside it."""
    return np.clip(x.ravel(), 0.05, 0.95)
