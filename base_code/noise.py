"""Noise generators for outcome draws. Each noise generator takes one covariate coordinate (a 1-D
array, shape (n,)) - pick the coordinate in the running script, e.g.

    noise_sampler = lambda x, rng: homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)
"""
import numpy as np


def _check_1d(x: np.ndarray) -> np.ndarray:
    """Fail loudly if a full covariate matrix is passed instead of a single coordinate x[:, j]."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected one covariate coordinate of shape (n,), e.g. x[:, 0]; got shape {x.shape}.")
    return x


def apply_noise(mu0: np.ndarray, mu1: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add noise draws to the noise-free potential outcomes, returning y0, y1."""
    return mu0 + e0, mu1 + e1


# --- Noise generators: draw e0, e1 to be passed into apply_noise ---


def homoskedastic_gaussian(x: np.ndarray, *, sd: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent N(0, sd) noise for e0 and e1, constant sd across all x."""
    x = _check_1d(x)
    n = x.shape[0]
    e0 = rng.normal(loc=0.0, scale=sd, size=n)
    e1 = rng.normal(loc=0.0, scale=sd, size=n)
    return e0, e1
