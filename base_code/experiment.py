"""Reusable experiment pipeline: fit a CATE learner on a simulated training sample, then
evaluate its estimated ATE against the true ATE on a test distribution. Shared by
running_code/main.py (a single run) and visualisation_code scripts (many runs swept over
some parameter, e.g. the test distribution's mean).
"""
import numpy as np

import noise
import propensity


def _mu1(x: np.ndarray, mu0: np.ndarray, *, mu1_shape=None, treatment_effect_shape=None) -> np.ndarray:
    """mu1(x) = mu1_shape(x) if given, else mu0 + treatment_effect_shape(x)."""
    if mu1_shape is not None:
        return mu1_shape(x)
    if treatment_effect_shape is not None:
        return mu0 + treatment_effect_shape(x)
    raise ValueError("Provide either mu1_shape or treatment_effect_shape.")


def simulate_training_outcomes(
    train_x: np.ndarray,
    *,
    mu0_shape,
    mu1_shape=None,
    treatment_effect_shape=None,
    noise_sampler,
    propensity_shape,
    rng: np.random.Generator,
):
    """Simulate training outcomes and treatment assignment from the given response-surface
    shapes, noise, and propensity. Returns (y_train, a_train) - split out from fit_learner so
    several learners can be fit and compared on the exact same simulated training data.
    """
    mu0_train = mu0_shape(train_x)
    mu1_train = _mu1(train_x, mu0_train, mu1_shape=mu1_shape, treatment_effect_shape=treatment_effect_shape)

    e0, e1 = noise_sampler(train_x, rng)
    y0_train, y1_train = noise.apply_noise(mu0_train, mu1_train, e0, e1)

    e_x = propensity_shape(train_x)
    a_train = propensity.get_assignment(e_x, rng=rng)
    y_train = np.where(a_train == 1, y1_train, y0_train)
    return y_train, a_train


def fit_learner(
    train_x: np.ndarray,
    *,
    mu0_shape,
    mu1_shape=None,
    treatment_effect_shape=None,
    noise_sampler,
    propensity_shape,
    learner,
    rng: np.random.Generator,
):
    """Simulate training outcomes from the given response-surface shapes, noise, and propensity,
    then fit learner on them. Returns (fitted learner, a_train) - a_train is the simulated
    treatment assignment, handed back so callers can e.g. color-code training samples by arm.
    """
    y_train, a_train = simulate_training_outcomes(
        train_x,
        mu0_shape=mu0_shape,
        mu1_shape=mu1_shape,
        treatment_effect_shape=treatment_effect_shape,
        noise_sampler=noise_sampler,
        propensity_shape=propensity_shape,
        rng=rng,
    )
    learner.fit(train_x, y_train, a_train)
    return learner, a_train


def true_tau(x: np.ndarray, *, mu0_shape=None, mu1_shape=None, treatment_effect_shape=None) -> np.ndarray:
    if treatment_effect_shape is not None:
        return treatment_effect_shape(x)
    if mu1_shape is not None:
        return mu1_shape(x) - mu0_shape(x)
    raise ValueError("Provide either mu1_shape or treatment_effect_shape.")


def evaluate(learner, test_x: np.ndarray, *, mu0_shape=None, mu1_shape=None, treatment_effect_shape=None) -> tuple[float, float, float]:
    """True mu0(x)/mu1(x) on the test distribution (no noise, no propensity: we know the truth),
    compared against the fitted learner's predictions. Returns (true_ate, estimated_ate, bias).
    """
    tau_test = true_tau(test_x, mu0_shape=mu0_shape, mu1_shape=mu1_shape, treatment_effect_shape=treatment_effect_shape)
    tau_hat_test = learner.predict(test_x)

    true = float(np.mean(tau_test))
    estimated = float(np.mean(tau_hat_test))
    bias = estimated - true
    return true, estimated, bias
