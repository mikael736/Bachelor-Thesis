"""Noise generators for outcome draws."""
import numpy as np


def apply_noise(mu0: np.ndarray, mu1: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add noise draws to the noise-free potential outcomes, returning y0, y1."""
    return mu0 + e0, mu1 + e1


# --- Noise generators: draw e0, e1 to be passed into apply_noise ---


def homoskedastic_gaussian(x: np.ndarray, *, sd: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent N(0, sd) noise for e0 and e1, constant sd across all x."""
    n = x.shape[0]
    e0 = rng.normal(loc=0.0, scale=sd, size=n)
    e1 = rng.normal(loc=0.0, scale=sd, size=n)
    return e0, e1
