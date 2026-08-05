"""Covariate (x) generators: sample the training and test/deployment distributions."""
import numpy as np

import config


def normal(n: int = config.DEFAULT_POPULATION_SIZE, *, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Normal(mean, sd). Returns shape (n, 1)."""
    return rng.normal(loc=mean, scale=sd, size=(n, 1))
