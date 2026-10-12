"""Comparison of plug-in ATE estimators built on the T-learner: the plain average of the CATE
predictions over the target vs. the AIPW-corrected estimate, with a 95% CI for each (naive for the
plain average). Design: experiment_design.md ("ATE-estimator comparison").
"""
import csv
import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np

import ate_estimators
import covariates
import noise
import propensity
import response_surface
import save_run

# -----------------------------------------------------------------------------
# 0. Run settings - part of the script, so the saved snapshot records them
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 100  # replications; each draws new source and target samples
TRAIN_POPULATION_SIZE = 1000  # source sample n
TARGET_SAMPLE_SIZE = 1000  # target sample m per sweep point, redrawn every rep
ACTUAL_ATE_SAMPLE_SIZE = 1_000_000  # per sweep point; actual ATE for multidimensional sweeps (exact in 1-D)
N_FOLDS = 5  # cross-fitting folds
MAX_DENSITY_RATIO = 20  # cap on the estimated density ratio (= 1 / PROPENSITY_CLIP; same meaning for any m)
PROPENSITY_CLIP = 0.05
# random-forest settings
# outcome forests mu0, mu1 and propensity forest e: as in metalearner_experiment.py
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)
# source-vs-target forest for the density ratio: leaf 100, chosen by OOB Brier score
# (tune_probability_forests.py) - its probabilities are divided by, leaf 5 is badly calibrated
DENSITY_RATIO_PARAMS = dict(max_features="sqrt", min_samples_leaf=100)

seed_seq = np.random.SeedSequence(SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(N_REPS)]
# separate stream for the actual-ATE sample, independent of the source and target samples
ate_rng = np.random.default_rng(seed_seq.spawn(1)[0])

# -----------------------------------------------------------------------------
# 1. Data-generating process (edit per experiment)
# -----------------------------------------------------------------------------

# source covariate distribution
train_covariates = lambda rng: covariates.beta(TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
mu0_shape = lambda x: response_surface.sine(x[:, 0], amplitude=3.0, frequency=2.0)
treatment_effect_shape = lambda x: response_surface.linear(x[:, 0], slope=7.0)
# jump locations of tau (e.g. [0.7] for a step), so the exact 1-D integral handles them; [] if continuous
treatment_effect_breakpoints = []

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)

# target-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

run_tag = "ate_estimators_experiment_4"

# -----------------------------------------------------------------------------
# 2. Each rep: new source and target samples, both estimators with their CIs per sweep point
# -----------------------------------------------------------------------------

estimates = {key: [] for key in ("plug_in", "se_plug_in", "aipw", "se_aipw", "ratio_capped_share")}
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{N_REPS}")
    x_train = train_covariates(rep_rng)
    mu0_train = mu0_shape(x_train)
    mu1_train = mu0_train + treatment_effect_shape(x_train)
    e0, e1 = noise_sampler(x_train, rep_rng)
    a_train = rep_rng.binomial(n=1, p=propensity_shape(x_train))
    y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)
    x_targets = test_dist.covariates(TARGET_SAMPLE_SIZE, rep_rng)

    rep_results = ate_estimators.transport_t_learner(
        x_train, y_train, a_train, x_targets,
        n_folds=N_FOLDS, propensity_clip=PROPENSITY_CLIP, max_density_ratio=MAX_DENSITY_RATIO,
        regressor_params=REGRESSOR_PARAMS, classifier_params=CLASSIFIER_PARAMS,
        density_ratio_params=DENSITY_RATIO_PARAMS, random_state=SEED,
    )
    for key in estimates:
        estimates[key].append([result[key] for result in rep_results])
estimates = {key: np.array(values) for key, values in estimates.items()}  # each (N_REPS, n_scenarios)

# actual ATE per scenario: exact in 1-D (numerical integration of tau against the test density; tau
# takes an (n, d) matrix, so the scalar x is wrapped as (1, 1)); otherwise a Monte Carlo average on a
# large independent sample, error SD = sd_t(tau) / sqrt(ACTUAL_ATE_SAMPLE_SIZE) - see experiment_design.md
if test_dist.distributions is not None:
    actual_ate = np.array([
        dist.expect(lambda x: treatment_effect_shape(np.array([[x]]))[0], points=treatment_effect_breakpoints or None)
        for dist in test_dist.distributions
    ])
else:
    actual_ate = np.array([
        treatment_effect_shape(ate_x).mean() for ate_x in test_dist.covariates(ACTUAL_ATE_SAMPLE_SIZE, ate_rng)
    ])

# per estimator: point estimates (for plot_experiment) and 95% CI coverage / mean width (for plot_coverage)
estimators = [("Plain average (naive CI)", "plug_in"), ("AIPW-corrected", "aipw")]  # both T-learner
results_list = []
for label, key in estimators:
    half_width = 1.96 * estimates[f"se_{key}"]
    results_list.append({
        "scenario_name": f"{run_tag}_{key}",
        "label": label,
        "test_distribution": test_dist.tick_labels,
        "axis_label": test_dist.axis_label,
        "estimated_ate_reps": estimates[key],
        "actual_ate": actual_ate,
        "ci_half_width_reps": half_width,
        "coverage": np.mean(np.abs(estimates[key] - actual_ate) <= half_width, axis=0),
        "mean_ci_width": np.mean(2 * half_width, axis=0),
    })

# -----------------------------------------------------------------------------
# 3. Save snapshot, summary table and plots to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_coverage, plot_experiment

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, SOURCE_SNAPSHOT, output_root=output_root)
plot_experiment(results_list, run_dir / "plot.png")
plot_coverage(results_list, N_REPS, run_dir / "ci_plot.png")

with open(run_dir / "summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["estimator", "test_distribution", "actual_ate", "mean_estimate", "mse", "coverage",
                     "mean_ci_width", "ratio_capped_share"])
    for results in results_list:
        mse = np.mean((results["estimated_ate_reps"] - actual_ate) ** 2, axis=0)
        mean_estimate = results["estimated_ate_reps"].mean(axis=0)
        for i, dist_label in enumerate(test_dist.labels):
            writer.writerow([results["label"], dist_label, actual_ate[i], mean_estimate[i], mse[i],
                             results["coverage"][i], results["mean_ci_width"][i],
                             estimates["ratio_capped_share"][:, i].mean()])
print(f"Saved to {run_dir}")
