"""Covariate (x) generators: sample the training and test/deployment distributions."""
import numpy as np


def normal(n: int, *, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Normal(mean, sd). Returns shape (n, 1)."""
    return rng.normal(loc=mean, scale=sd, size=(n, 1))


def beta(n: int, *, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Beta(a, b), supported on [0, 1]. Returns shape (n, 1)."""
    return rng.beta(a, b, size=(n, 1))
