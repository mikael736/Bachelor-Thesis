"""Shared reader for the summary.csv files of the simulation experiments, used by
meta_learner_comparison.py and ate_estimator_comparison.py.

load(experiment) returns, per experiment number 1-4, a dict {(method, target): Stats}; method is the value of
the column "learner" (meta-learner experiments) or "estimator" (ATE-estimator experiments), target e.g.
"Beta(2,8)". Bias, variance and the Monte Carlo standard errors (MCSE) follow Appendix "Bias and MSE with
Monte Carlo standard errors": the variance is MSE - bias^2 (divisor N_REPS), the MCSE of the bias is the sd of
the estimates (divisor N_REPS - 1) / sqrt(N_REPS), the MCSE of the MSE is the column mcse_mse.
"""
import csv
import math
from dataclasses import dataclass
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[2] / "visualisation_output"
N_REPS = 100
TARGETS = [f"Beta({a},{10 - a})" for a in range(2, 9)]  # sweep order
CENTER, ENDS = "Beta(5,5)", ["Beta(2,8)", "Beta(8,2)"]


@dataclass
class Stats:
    mean_estimate: float
    bias: float
    variance: float
    mse: float
    mcse_mse: float
    mcse_bias: float
    coverage: float | None = None  # ATE-estimator experiments only
    mean_ci_width: float | None = None


def load(experiment):
    """experiment: "metalearner_experiment" or "ate_estimator_experiment"."""
    key = "learner" if experiment == "metalearner_experiment" else "estimator"
    results = {}
    for exp_num in range(1, 5):
        with open(OUTPUTS / f"{experiment}_{exp_num}" / "summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        stats = {}
        for r in rows:
            mse, bias = float(r["mse"]), float(r["mean_estimate"]) - float(r["actual_ate"])
            variance = mse - bias**2
            stats[r[key], r["test_distribution"]] = Stats(
                mean_estimate=float(r["mean_estimate"]), bias=bias, variance=variance, mse=mse,
                mcse_mse=float(r["mcse_mse"]),
                mcse_bias=math.sqrt(variance * N_REPS / (N_REPS - 1) / N_REPS),
                coverage=float(r["coverage"]) if "coverage" in r else None,
                mean_ci_width=float(r["mean_ci_width"]) if "mean_ci_width" in r else None,
            )
        results[exp_num] = stats
    return results


def detectable(a, b):
    """Section "Plug-in estimates of the meta-learners": a difference of two MSEs is detectable if it exceeds
    twice the sum of their MCSEs."""
    return abs(a.mse - b.mse) > 2 * (a.mcse_mse + b.mcse_mse)
