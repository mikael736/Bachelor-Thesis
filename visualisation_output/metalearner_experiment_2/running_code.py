"""Meta-learner comparison: plug-in ATE estimates of the T-, X- and DR-learner along a sweep of target
distributions, for one experiment E1-E4 set in part 1.
"""
import csv
import sys
from pathlib import Path

# snapshot of this script at launch, saved with the outputs
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np

import cate_learners
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
TEST_POPULATION_SIZE = 100_000  # target sample size per target distribution
ACTUAL_ATE_SAMPLE_SIZE = 1_000_000  # per target distribution; target ATE for multidimensional sweeps (numerical integration in 1-D)
# random-forest settings (max_features has no effect with one covariate; leaf size: metalearner_tuning.py)
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)

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
treatment_effect_shape = lambda x: response_surface.linear(x[:, 0], slope=7.0)
# jump locations of tau (e.g. [0.7] for a step), so the exact 1-D integral handles them; [] if continuous
treatment_effect_breakpoints = []

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)

# CATE learners to fit and compare
rf = dict(base_learner="rf", regressor_params=REGRESSOR_PARAMS, random_state=SEED)
learners = [
    ("T-learner", lambda rng: cate_learners.TLearner(**rf)),
    ("X-learner", lambda rng: cate_learners.XLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
    ("DR-learner", lambda rng: cate_learners.DRLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
]

# target-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output directory, one per experiment
run_tag = "metalearner_experiment_2"

# -----------------------------------------------------------------------------
# 2. Each replication: a new study sample and new target samples, shared by all learners
# -----------------------------------------------------------------------------

estimated_ate_reps = {tag: [] for tag, _ in learners}
# variance of the CATE predictions over each target sample (target sampling error)
prediction_var_reps = {tag: [] for tag, _ in learners}
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{N_REPS}")
    x_train = train_covariates(rep_rng)
    mu0_train = mu0_shape(x_train)
    mu1_train = mu0_train + treatment_effect_shape(x_train)
    e0, e1 = noise_sampler(x_train, rep_rng)
    a_train = rep_rng.binomial(n=1, p=propensity_shape(x_train))
    y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)
    # target samples drawn after the study sample, so the study sample per replication is unchanged
    test_scenarios = test_dist.covariates(TEST_POPULATION_SIZE, rep_rng)

    for tag, make_learner in learners:
        fitted_learner = make_learner(rep_rng).fit(x_train, y_train, a_train)
        predictions = [fitted_learner.predict(test_x) for test_x in test_scenarios]
        estimated_ate_reps[tag].append([pred.mean() for pred in predictions])
        prediction_var_reps[tag].append([pred.var(ddof=1) for pred in predictions])

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

results_list = [
    {
        "scenario_name": f"{run_tag}_{tag}",
        "label": tag,
        "test_distribution": test_dist.tick_labels,
        "axis_label": test_dist.axis_label,
        "estimated_ate_reps": np.array(estimated_ate_reps[tag]),  # shape (N_REPS, len(test_scenarios))
        "prediction_var_reps": np.array(prediction_var_reps[tag]),  # same shape
        "actual_ate": actual_ate,
    }
    for tag, _ in learners
]

# -----------------------------------------------------------------------------
# 3. Save snapshot, summary table and plot to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_experiment

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, SOURCE_SNAPSHOT, output_root=output_root)
plot_experiment(results_list, run_dir / "plot.png")

with open(run_dir / "summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["learner", "test_distribution", "actual_ate", "mean_estimate", "mse", "mcse_mse",
                     "mean_prediction_var"])
    for results in results_list:
        squared_errors = (results["estimated_ate_reps"] - actual_ate) ** 2
        mse = squared_errors.mean(axis=0)
        # Monte Carlo standard error of the MSE: sd of the squared errors over replications / sqrt(N_REPS)
        mcse_mse = squared_errors.std(axis=0, ddof=1) / np.sqrt(N_REPS)
        mean_estimate = results["estimated_ate_reps"].mean(axis=0)
        mean_prediction_var = results["prediction_var_reps"].mean(axis=0)
        for i, dist_label in enumerate(test_dist.labels):
            writer.writerow([results["label"], dist_label, actual_ate[i], mean_estimate[i], mse[i], mcse_mse[i],
                             mean_prediction_var[i]])

# per-replication estimates (N_REPS, n_targets) per learner
np.savez(run_dir / "estimates.npz", actual_ate=actual_ate,
         **{r["label"]: r["estimated_ate_reps"] for r in results_list})
print(f"Saved to {run_dir}")
