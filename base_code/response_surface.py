"""Response surfaces: mu0(x) and mu1(x), the noise-free potential outcomes."""
import numpy as np


def linear_homogeneous(x: np.ndarray, *, intercept: float, slope: float, treatment_effect: float) -> tuple[np.ndarray, np.ndarray]:
    """mu0(x) = intercept + slope * x; mu1(x) = mu0(x) + treatment_effect (CATE(x) == treatment_effect for every x)."""
    mu0 = intercept + slope * x[:, 0]
    mu1 = mu0 + treatment_effect
    return mu0, mu1
