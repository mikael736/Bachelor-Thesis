"""Covariate generators for the study and target distributions (part 1) and named sweeps of target
distributions built on them (part 2), selected by name in the running scripts.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


def normal(n: int, *, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Normal(mean, sd). Returns shape (n, 1)."""
    return rng.normal(loc=mean, scale=sd, size=(n, 1))


def beta(n: int, *, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws from a 1-D Beta(a, b), supported on [0, 1]. Returns shape (n, 1)."""
    return rng.beta(a, b, size=(n, 1))


def coordinatewise_3d(n: int, *, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n draws of a 3-D covariate vector (x0, x1, x2), each coordinate defined separately
    (not a joint 3-D distribution), with
        x0 ~ Beta(a, b)
        x1 = x0^2          (fully dependent on x0)
        x2 ~ Beta(2, 8)      (independent of x0 and of (a, b))
    Returns shape (n, 3), one row per unit and one column per coordinate (column j is xj).
    """
    x0 = rng.beta(a, b, size=(n, 1))
    x1 = x0**2
    x2 = rng.beta(2, 8, size=(n, 1))
    return np.hstack([x0, x1, x2])


# -----------------------------------------------------------------------------
# Part 2: named test-distribution sweeps
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDistribution:
    """One named sweep of target populations. labels names each sweep point in full (tables).
    covariates(n, rng) draws a fresh sample of n units for every sweep point and returns them as a
    list, in the same order as labels. distributions (1-D sweeps only) holds the scipy distribution
    of each sweep point, for exact expectations; None for multidimensional sweeps. For plots,
    tick_labels are short per-point labels and axis_label names the family on the axis.
    """

    labels: list[str]
    covariates: Callable[[int, np.random.Generator], list[np.ndarray]]
    tick_labels: list[str]
    axis_label: str
    distributions: list | None = None


_normal_mean_sweep_means = np.linspace(-1.0, 1.0, 5)
NORMAL_MEAN_SWEEP = TestDistribution(
    labels=[f"N({mean:g},1)" for mean in _normal_mean_sweep_means],
    covariates=lambda n, rng: [
        normal(n, mean=mean, sd=1.0, rng=rng) for mean in _normal_mean_sweep_means
    ],
    tick_labels=[f"{mean:g}" for mean in _normal_mean_sweep_means],
    axis_label="Target distribution N(mean, 1): mean",
    distributions=[stats.norm(loc=mean, scale=1.0) for mean in _normal_mean_sweep_means],
)


_beta_shape_sweep_a = np.linspace(2, 8, 7)
_beta_shape_sweep_params = [(a, 10.0 - a) for a in _beta_shape_sweep_a]
BETA_SHAPE_SWEEP = TestDistribution(
    labels=[f"Beta({a:g},{b:g})" for a, b in _beta_shape_sweep_params],
    covariates=lambda n, rng: [
        beta(n, a=a, b=b, rng=rng) for a, b in _beta_shape_sweep_params
    ],
    tick_labels=[f"({a:g},{b:g})" for a, b in _beta_shape_sweep_params],
    axis_label=r"Target distribution Beta($\alpha$, $\beta$)",
    distributions=[stats.beta(a, b) for a, b in _beta_shape_sweep_params],
)


# same (a, b) sweep as BETA_SHAPE_SWEEP, but on coordinatewise_3d: only x0 (and hence x1) shifts, x2 stays fixed
COORDINATEWISE_3D_SHAPE_SWEEP = TestDistribution(
    labels=[f"x0~Beta({a:g},{b:g})" for a, b in _beta_shape_sweep_params],
    covariates=lambda n, rng: [
        coordinatewise_3d(n, a=a, b=b, rng=rng) for a, b in _beta_shape_sweep_params
    ],
    tick_labels=[f"({a:g},{b:g})" for a, b in _beta_shape_sweep_params],
    axis_label=r"Target distribution of $x_0$: Beta($\alpha$, $\beta$)",
)


TEST_DISTRIBUTIONS = {
    "normal-mean-sweep": NORMAL_MEAN_SWEEP,
    "beta-shape-sweep": BETA_SHAPE_SWEEP,
    "coordinatewise-3d-shape-sweep": COORDINATEWISE_3D_SHAPE_SWEEP,
}
