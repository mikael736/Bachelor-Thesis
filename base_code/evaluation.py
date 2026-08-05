"""Evaluation: compare the true ATE (from tau) against the learner's estimated ATE (from tau_hat)."""
import numpy as np


def true_ate(tau: np.ndarray) -> float:
    """True ATE: mean of tau = mu1 - mu0 over the test distribution."""
    return float(np.mean(tau))


def estimated_ate(tau_hat: np.ndarray) -> float:
    """Estimated ATE: mean of the CATE learner's predictions over the test distribution."""
    return float(np.mean(tau_hat))


def ate_bias(true_ate: float, estimated_ate: float) -> float:
    """Signed difference between the estimated and true ATE."""
    return estimated_ate - true_ate
