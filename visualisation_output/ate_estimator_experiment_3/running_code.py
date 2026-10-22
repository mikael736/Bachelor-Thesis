"""Comparison of two ATE estimators built on the cross-fitted T-learner: the plug-in estimate (average
of the CATE predictions over the target) vs. the AIPW-corrected estimate, with a 95% CI for each (naive
for the plug-in: target sampling only), for one experiment E1-E4 set in part 1.
"""
import csv
import sys
from pathlib import Path

# snapshot of this script at launch, saved with the outputs
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
N_REPS = 100  # replications; each draws a new study sample and new target samples
TRAIN_POPULATION_SIZE = 1000  # study sample size n
TARGET_SAMPLE_SIZE = 1000  # target sample size m per target distribution
ACTUAL_ATE_SAMPLE_SIZE = 1_000_000  # per target distribution; target ATE for multidimensional sweeps (numerical integration in 1-D)
N_FOLDS = 5  # cross-fitting folds
MAX_DENSITY_RATIO = 20  # cap on the estimated density ratio
PROPENSITY_CLIP = 0.05
# random-forest settings
# response surfaces mu0, mu1 and propensity score e: as in metalearner_experiment.py
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)
# study-vs-target forest for the density ratio (leaf size: ate_estimator_tuning.py)
DENSITY_RATIO_PARAMS = dict(max_features="sqrt", min_samples_leaf=100)

seed_seq = np.random.SeedSequence(SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(N_REPS)]
# separate stream for the target-ATE sample, independent of the study and target samples
ate_rng = np.random.default_rng(seed_seq.spawn(1)[0])

# -----------------------------------------------------------------------------
# 1. Data-generating process (set per experiment)
# -----------------------------------------------------------------------------

# study covariate distribution
train_covariates = lambda rng: covariates.beta(TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
# E1: mu0 = sine(amplitude=3, frequency=2), tau = constant(1)
# E2: mu0 = linear(slope=7), tau = linear(slope=7)
# E3: mu0 = linear(slope=7), tau = step(jump=3, threshold=0.7), breakpoints [0.7]
# E4: mu0 = sine(amplitude=3, frequency=2), tau = linear(slope=7)
mu0_shape = lambda x: response_surface.linear(x[:, 0], slope=7.0)
treatment_effect_shape = lambda x: response_surface.step(x[:, 0], jump=3.0, threshold=0.7)
# jump locations of tau (e.g. [0.7] for a step), so the exact 1-D integral handles them; [] if continuous
treatment_effect_breakpoints = [0.7]

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)

# target-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output directory, one per experiment
run_tag = "ate_estimator_experiment_3"

# -----------------------------------------------------------------------------
# 2. Each replication: a new study sample and new target samples, both estimators with their CIs
# -----------------------------------------------------------------------------

estimates = {key: [] for key in ("plug_in", "se_plug_in", "aipw", "se_aipw", "ratio_capped_share",
                                 # for the variance diagnostic (diagnostics.csv)
                                 "t_bar", "q_bar", "var_t_bar", "var_q_bar", "se_aipw_oracle_resid")}
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
        # true response surfaces, only for the variance diagnostic with the true residuals
        mu0_true=mu0_shape, mu1_true=lambda x: mu0_shape(x) + treatment_effect_shape(x),
    )
    for key in estimates:
        estimates[key].append([result[key] for result in rep_results])
estimates = {key: np.array(values) for key, values in estimates.items()}  # each (N_REPS, n_scenarios)

# target ATE per target distribution: in 1-D by numerical integration of tau against the target density
# (tau takes an (n, d) matrix, so the scalar x is wrapped as (1, 1)); otherwise a Monte Carlo average on
# a large independent sample
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
estimators = [("Plug-in (naive CI)", "plug_in"), ("AIPW-corrected", "aipw")]  # both T-learner
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
    writer.writerow(["estimator", "test_distribution", "actual_ate", "mean_estimate", "mse", "mcse_mse",
                     "coverage", "mean_ci_width", "ratio_capped_share"])
    for results in results_list:
        squared_errors = (results["estimated_ate_reps"] - actual_ate) ** 2
        mse = squared_errors.mean(axis=0)
        # Monte Carlo standard error of the MSE: sd of the squared errors over replications / sqrt(N_REPS)
        mcse_mse = squared_errors.std(axis=0, ddof=1) / np.sqrt(N_REPS)
        mean_estimate = results["estimated_ate_reps"].mean(axis=0)
        for i, dist_label in enumerate(test_dist.labels):
            writer.writerow([results["label"], dist_label, actual_ate[i], mean_estimate[i], mse[i], mcse_mse[i],
                             results["coverage"][i], results["mean_ci_width"][i],
                             estimates["ratio_capped_share"][:, i].mean()])

# variance diagnostic of the AIPW-corrected estimate:
# - empirical vs. estimated variance of the estimate and of its two parts t_bar and q_bar
#   (var = variance across replications, est = mean over replications of the estimated variance);
#   cov_t_q is the empirical covariance of t_bar and q_bar, which the variance estimate ignores
# - the estimated variance with q from the true residuals y - mu_a(x), and the resulting coverage
with open(run_dir / "diagnostics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_distribution", "var_aipw", "est_var_aipw", "var_t_bar", "est_var_t_bar",
                     "var_q_bar", "est_var_q_bar", "cov_t_q", "est_var_aipw_oracle_resid",
                     "coverage_aipw", "coverage_aipw_oracle_resid"])
    var_aipw = estimates["aipw"].var(axis=0, ddof=1)
    var_t_bar = estimates["t_bar"].var(axis=0, ddof=1)
    var_q_bar = estimates["q_bar"].var(axis=0, ddof=1)
    cov_t_q = np.array([np.cov(estimates["t_bar"][:, i], estimates["q_bar"][:, i])[0, 1]
                        for i in range(len(actual_ate))])
    se_oracle = estimates["se_aipw_oracle_resid"]
    coverage_oracle = np.mean(np.abs(estimates["aipw"] - actual_ate) <= 1.96 * se_oracle, axis=0)
    coverage_aipw = results_list[1]["coverage"]
    for i, dist_label in enumerate(test_dist.labels):
        writer.writerow([dist_label, var_aipw[i], (estimates["se_aipw"][:, i] ** 2).mean(),
                         var_t_bar[i], estimates["var_t_bar"][:, i].mean(),
                         var_q_bar[i], estimates["var_q_bar"][:, i].mean(), cov_t_q[i],
                         (se_oracle[:, i] ** 2).mean(), coverage_aipw[i], coverage_oracle[i]])

# per-replication estimates and standard errors (N_REPS, n_targets)
np.savez(run_dir / "estimates.npz", actual_ate=actual_ate, **estimates)
print(f"Saved to {run_dir}")
