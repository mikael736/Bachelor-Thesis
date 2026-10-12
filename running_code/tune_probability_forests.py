"""Leaf size of the two probability forests of ate_estimator_experiment.py, whose outputs enter the
correction as weights: the propensity forest e (source) and the source-vs-target forest rho (pooled
source + target, per sweep point). Selection rule: out-of-bag (OOB) Brier score, observable in
practice. Validation (simulation only): error against the true e and rho, and of the capped
density ratio r against the true r at the source points, where it enters the correction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

import covariates
import propensity

SEED = 42
N_SEEDS = 20
SOURCE_SIZE = 1000
TARGET_SIZE = 1000
MAX_DENSITY_RATIO = 20
LEAF_SIZES = [5, 10, 20, 50, 100, 200]

# data-generating process - keep in sync with ate_estimator_experiment.py
source_covariates = lambda rng: covariates.beta(SOURCE_SIZE, a=5.0, b=5.0, rng=rng)
source_density = lambda x: stats.beta(5.0, 5.0).pdf(x[:, 0])
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)
test_dist = covariates.TEST_DISTRIBUTIONS["beta-shape-sweep"]


def oob_forest(x, labels, leaf):
    """OOB probability of class 1 (nan for the rare units that are never out of bag)."""
    clf = RandomForestClassifier(max_features="sqrt", min_samples_leaf=leaf, oob_score=True, random_state=SEED)
    with np.errstate(invalid="ignore", divide="ignore"):
        return clf.fit(x, labels).oob_decision_function_[:, 1]


rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(N_SEEDS)]
n_sweep = len(test_dist.labels)
e_scores = np.zeros((len(LEAF_SIZES), 2))  # (Brier, rmse vs. true e)
rho_scores = np.zeros((len(LEAF_SIZES), n_sweep, 4))  # (Brier, rmse vs. true rho, rmse of capped r, capped share)

for rng in rngs:
    x_s = source_covariates(rng)
    e_true = propensity_shape(x_s)
    a = rng.binomial(n=1, p=e_true)
    x_targets = test_dist.covariates(TARGET_SIZE, rng)
    for i, leaf in enumerate(LEAF_SIZES):
        e_hat = oob_forest(x_s, a, leaf)
        e_scores[i] += [np.nanmean((e_hat - a) ** 2), np.sqrt(np.nanmean((e_hat - e_true) ** 2))]
        for k, (x_t, dist) in enumerate(zip(x_targets, test_dist.distributions)):
            x_pool = np.vstack([x_s, x_t])
            s = np.r_[np.ones(SOURCE_SIZE, dtype=int), np.zeros(TARGET_SIZE, dtype=int)]
            p_s, p_t = source_density(x_pool), dist.pdf(x_pool[:, 0])
            rho_true = SOURCE_SIZE * p_s / (SOURCE_SIZE * p_s + TARGET_SIZE * p_t)
            rho_hat = oob_forest(x_pool, s, leaf)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_hat = SOURCE_SIZE / TARGET_SIZE * (1 - rho_hat[:SOURCE_SIZE]) / rho_hat[:SOURCE_SIZE]
            r_true = p_t[:SOURCE_SIZE] / p_s[:SOURCE_SIZE]
            rho_scores[i, k] += [
                np.nanmean((rho_hat - s) ** 2),
                np.sqrt(np.nanmean((rho_hat - rho_true) ** 2)),
                np.sqrt(np.nanmean((np.minimum(r_hat, MAX_DENSITY_RATIO) - np.minimum(r_true, MAX_DENSITY_RATIO)) ** 2)),
                np.nanmean(r_hat > MAX_DENSITY_RATIO),
            ]
e_scores /= N_SEEDS
rho_scores /= N_SEEDS

print(f"Propensity forest e (n = {SOURCE_SIZE}, {N_SEEDS} seeds); * = best")
best = e_scores.argmin(axis=0)
print(f"{'leaf':>6}{'OOB Brier':>12}{'rmse vs e':>12}")
for i, leaf in enumerate(LEAF_SIZES):
    print(f"{leaf:>6}" + "".join(f"{v:>11.4f}{'*' if i == best[j] else ' '}" for j, v in enumerate(e_scores[i])))

print(f"\nDensity-ratio forest rho ({SOURCE_SIZE} source + {TARGET_SIZE} target), mean over the sweep; * = best")
mean_rho = rho_scores.mean(axis=1)
best = mean_rho.argmin(axis=0)
print(f"{'leaf':>6}{'OOB Brier':>12}{'rmse vs rho':>12}{'rmse r capped':>14}{'capped share':>14}")
for i, leaf in enumerate(LEAF_SIZES):
    print(f"{leaf:>6}" + "".join(f"{v:>11.4f}{'*' if i == best[j] and j < 3 else ' '}" for j, v in enumerate(mean_rho[i][:3]))
          + f"{mean_rho[i][3]:>14.4f}")

print("\nOOB Brier of rho per sweep point")
print(f"{'leaf':>6}" + "".join(f"{label:>15}" for label in test_dist.labels))
best = rho_scores[:, :, 0].argmin(axis=0)
for i, leaf in enumerate(LEAF_SIZES):
    print(f"{leaf:>6}" + "".join(f"{rho_scores[i, k, 0]:>14.4f}{'*' if i == best[k] else ' '}" for k in range(n_sweep)))
