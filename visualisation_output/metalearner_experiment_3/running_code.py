import csv
import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0,str(Path(__file__).resolve().parent.parent / "base_code"))

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
N_REPS = 100  # training/evaluation replications per learner and test scenario
TRAIN_POPULATION_SIZE = 1000
TEST_POPULATION_SIZE = 100_000  # per test-distribution sweep point, redrawn every rep
ACTUAL_ATE_SAMPLE_SIZE = 1_000_000  # per sweep point; actual ATE for multidimensional sweeps (exact in 1-D)
# random-forest settings (leaf size 5 chosen under covariate shift; max_features: standard p/3 for
# regression and sqrt(p) for classification, no effect in 1-D - see experiment_design.md)
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)

seed_seq = np.random.SeedSequence(SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(N_REPS)]
# separate stream for the actual-ATE sample, independent of the training and test samples
ate_rng = np.random.default_rng(seed_seq.spawn(1)[0])

# -----------------------------------------------------------------------------
# 1. Training setup (edit per experiment)
# -----------------------------------------------------------------------------

# training covariate distribution
train_covariates = lambda rng: covariates.beta(TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
mu0_shape = lambda x: response_surface.linear(x[:, 0], slope=7.0)
treatment_effect_shape = lambda x: response_surface.step(x[:, 0], jump=3.0, threshold=0.7)
# jump locations of tau (e.g. [0.7] for a step), so the exact 1-D integral handles them; [] if continuous
treatment_effect_breakpoints = [0.7]

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

# test-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output dir name - keep it current when you edit the setup above
run_tag = "metalearner_experiment_3"

# -----------------------------------------------------------------------------
# 2. Evaluate each learner on each test scenario, over N_REPS replications. Each replication draws
# new training data and new target samples, shared across all learners for a paired comparison -
# only the learner differs, not the data.
# -----------------------------------------------------------------------------

estimated_ate_reps = {tag: [] for tag, _ in learners}
# variance of the CATE predictions over each target sample: the target sampling error adds its mean / M
# to the expected MSE (experiment_design.md, part B)
prediction_var_reps = {tag: [] for tag, _ in learners}
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{N_REPS}")
    x_train = train_covariates(rep_rng)
    mu0_train = mu0_shape(x_train)
    mu1_train = mu0_train + treatment_effect_shape(x_train)
    e0, e1 = noise_sampler(x_train, rep_rng)
    a_train = rep_rng.binomial(n=1, p=propensity_shape(x_train))
    y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)
    # target samples drawn after the training data, so the training data per rep is unchanged
    test_scenarios = test_dist.covariates(TEST_POPULATION_SIZE, rep_rng)

    for tag, make_learner in learners:
        fitted_learner = make_learner(rep_rng).fit(x_train, y_train, a_train)
        predictions = [fitted_learner.predict(test_x) for test_x in test_scenarios]
        estimated_ate_reps[tag].append([pred.mean() for pred in predictions])
        prediction_var_reps[tag].append([pred.var(ddof=1) for pred in predictions])

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
        # Monte Carlo standard error of the MSE: sd of the squared errors over reps / sqrt(N_REPS)
        mcse_mse = squared_errors.std(axis=0, ddof=1) / np.sqrt(N_REPS)
        mean_estimate = results["estimated_ate_reps"].mean(axis=0)
        mean_prediction_var = results["prediction_var_reps"].mean(axis=0)
        for i, dist_label in enumerate(test_dist.labels):
            writer.writerow([results["label"], dist_label, actual_ate[i], mean_estimate[i], mse[i], mcse_mse[i],
                             mean_prediction_var[i]])

# per-rep estimates (N_REPS, n_scenarios) per learner, for paired comparisons across learners later
np.savez(run_dir / "estimates.npz", actual_ate=actual_ate,
         **{r["label"]: r["estimated_ate_reps"] for r in results_list})
print(f"Saved to {run_dir}")
