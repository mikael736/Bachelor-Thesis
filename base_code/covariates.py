"""Part 1: covariate (x) generators - sample the training and test/deployment distributions.
Part 2: named test-distribution sweeps built on top of them - each is a family of test
populations (e.g. "Normal(mean,1) for mean sweeping -5..5") that a running_code script picks by
name to evaluate a fitted learner's extrapolation as the test population drifts from training.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import config


def normal(n: int, *, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Normal(mean, sd). Returns shape (n, 1)."""
    return rng.normal(loc=mean, scale=sd, size=(n, 1))


def beta(n: int, *, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Beta(a, b), supported on [0, 1]. Returns shape (n, 1)."""
    return rng.beta(a, b, size=(n, 1))


# -----------------------------------------------------------------------------
# Part 2: named test-distribution sweeps
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDistribution:
    """One named sweep of test populations. labels/positions are its plot-facing description
    (a string and a numeric x-axis position per sweep point) - positions is just "where this
    point sits on the x-axis", not necessarily the distribution's statistical mean (it happens
    to coincide for some sweeps, e.g. beta_shape_sweep1, but that's not guaranteed in general).
    covariates(rng), called once per replication, draws a fresh sample for every sweep point and
    returns them as a list, in the same order as labels/positions.
    """

    labels: list[str]
    positions: np.ndarray
    covariates: Callable[[np.random.Generator], list[np.ndarray]]


_normal_mean_sweep1_means = np.linspace(-5.0, 5.0, 11)
NORMAL_MEAN_SWEEP1 = TestDistribution(
    labels=[f"N({mean:.1f},1.0)" for mean in _normal_mean_sweep1_means],
    positions=_normal_mean_sweep1_means,
    covariates=lambda rng: [
        normal(config.TEST_POPULATION_SIZE, mean=mean, sd=1.0, rng=rng) for mean in _normal_mean_sweep1_means
    ],
)


_beta_shape_sweep1_a = np.linspace(1.5, 8.5, 11)
_beta_shape_sweep1_params = [(a, 10.0 - a) for a in _beta_shape_sweep1_a]
BETA_SHAPE_SWEEP1 = TestDistribution(
    labels=[f"Beta({a:.1f},{b:.1f})" for a, b in _beta_shape_sweep1_params],
    positions=_beta_shape_sweep1_a / 10.0,
    covariates=lambda rng: [
        beta(config.TEST_POPULATION_SIZE, a=a, b=b, rng=rng) for a, b in _beta_shape_sweep1_params
    ],
)


TEST_DISTRIBUTIONS = {
    "normal_mean_sweep1": NORMAL_MEAN_SWEEP1,
    "beta_shape_sweep1": BETA_SHAPE_SWEEP1,
}
