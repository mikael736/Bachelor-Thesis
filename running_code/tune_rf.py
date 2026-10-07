"""Compare random-forest min_samples_leaf values by out-of-bag (OOB) error of the nuisance models
mu0, mu1 (regressors, per arm) and e (classifier), on the same data-generating process as
current_experiment.py. Reports OOB error against the observed y / a and, since the data is
simulated, against the true mu0, mu1, e (no noise floor).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import covariates
import noise
import propensity
import response_surface

SEED = 42
N_REPS = 20
TRAIN_POPULATION_SIZE = 1000
LEAF_SIZES = [1, 2, 5, 10, 20, 50, 100]

# data-generating process - keep in sync with current_experiment.py
train_covariates = lambda rng: covariates.beta(TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)
mu0_shape = lambda x: response_surface.sine(x[:, 0], amplitude=3.0, frequency=2.0)
treatment_effect_shape = lambda x: response_surface.constant(x[:, 0], value=2.0)
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)

# errors[leaf][metric] -> list over reps
metrics = ["mu0 vs y", "mu0 vs truth", "mu1 vs y", "mu1 vs truth", "e Brier vs a", "e vs truth"]
errors = {leaf: {m: [] for m in metrics} for leaf in LEAF_SIZES}

for rep_rng in [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(N_REPS)]:
    x = train_covariates(rep_rng)
    mu0 = mu0_shape(x)
    mu1 = mu0 + treatment_effect_shape(x)
    e0, e1 = noise_sampler(x, rep_rng)
    e = propensity_shape(x)
    a = rep_rng.binomial(n=1, p=e)
    y = np.where(a == 1, mu1 + e1, mu0 + e0)

    for leaf in LEAF_SIZES:
        for arm, mu in [(0, mu0), (1, mu1)]:
            rf = RandomForestRegressor(min_samples_leaf=leaf, oob_score=True, random_state=SEED)
            rf.fit(x[a == arm], y[a == arm])
            errors[leaf][f"mu{arm} vs y"].append(np.mean((rf.oob_prediction_ - y[a == arm]) ** 2))
            errors[leaf][f"mu{arm} vs truth"].append(np.mean((rf.oob_prediction_ - mu[a == arm]) ** 2))

        clf = RandomForestClassifier(min_samples_leaf=leaf, oob_score=True, random_state=SEED).fit(x, a)
        e_oob = clf.oob_decision_function_[:, 1]
        errors[leaf]["e Brier vs a"].append(np.mean((e_oob - a) ** 2))
        errors[leaf]["e vs truth"].append(np.mean((e_oob - e) ** 2))

# mean OOB MSE over reps; best value per column marked with *
means = {m: np.array([np.mean(errors[leaf][m]) for leaf in LEAF_SIZES]) for m in metrics}
print(f"{'leaf':>5}" + "".join(f"{m:>15}" for m in metrics))
for i, leaf in enumerate(LEAF_SIZES):
    cells = [f"{means[m][i]:.4f}" + ("*" if i == means[m].argmin() else " ") for m in metrics]
    print(f"{leaf:>5}" + "".join(f"{c:>15}" for c in cells))
