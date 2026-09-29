import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0,str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np
from scipy import stats

import cate_learners
import covariates
import experiment
import noise
import propensity
import response_surface
import save_run

# -----------------------------------------------------------------------------
# 0. Run settings - part of the script, so the saved snapshot records them
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 50  # training/evaluation replications averaged into each learner's mean ATE bias
TRAIN_POPULATION_SIZE = 1000
TEST_POPULATION_SIZE = 20_000  # per test-distribution sweep point

seed_seq = np.random.SeedSequence(SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(N_REPS)]
# fixed target sample, drawn once and reused across all reps
test_rng = np.random.default_rng(SEED)

# -----------------------------------------------------------------------------
# 1. Training setup (edit per experiment)
# -----------------------------------------------------------------------------

# training covariate distribution
population_tag = "beta5-5"
train_covariates = lambda rng: covariates.beta(TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
mu0_tag = "exponential"
mu0_shape = lambda x: response_surface.exponential(x[:, 0], multiplier=4.0)
mu1_tag = "mu0+constant"
treatment_effect_shape = lambda x: response_surface.constant(x[:, 0], value=100.0)

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_tag = "inv-linear"
propensity_shape = lambda x: propensity.linear(1 - x[:, 0])

# CATE learners to fit and compare
learners = [
    ("XLearnerRF", lambda rng: cate_learners.XLearner(base_learner="rf", random_state=SEED)),
    ("TLearnerRF", lambda rng: cate_learners.TLearner(base_learner="rf", random_state=SEED)),
    ("DRLearnerRF", lambda rng: cate_learners.DRLearner(base_learner="rf", random_state=SEED)),
]

# test-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep1" 
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output dir name, composed from the *_tag values above - keep tags current when you edit them
learner_tag = "-vs-".join(tag for tag, _ in learners)
run_tag = f"{population_tag}_{mu0_tag}_{mu1_tag}_{propensity_tag}"

# -----------------------------------------------------------------------------
# 2. Evaluate each learner on each test scenario, over N_REPS replications. Each
# replication's training data is shared across all learners for a paired comparison - only the
# learner differs, not the data.
# -----------------------------------------------------------------------------

test_scenarios = test_dist.covariates(TEST_POPULATION_SIZE, test_rng)

bias_reps = {tag: [] for tag, _ in learners}
fitted_learners_last, train_x_last, train_a_last = None, None, None
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{N_REPS}")
    train_x = train_covariates(rep_rng)
    y_train, a_train = experiment.simulate_training_outcomes(
        train_x,
        mu0_shape=mu0_shape,
        treatment_effect_shape=treatment_effect_shape,
        noise_sampler=noise_sampler,
        propensity_shape=propensity_shape,
        rng=rep_rng,
    )

    fitted_learners_last = {}
    for tag, make_learner in learners:
        fitted_learner = make_learner(rep_rng).fit(train_x, y_train, a_train)
        bias_reps[tag].append([
            experiment.evaluate(fitted_learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
            for test_x in test_scenarios
        ])
        fitted_learners_last[tag] = fitted_learner

    train_x_last, train_a_last = train_x, a_train

# CATE panel: true vs. predicted tau(x), evaluated at the last rep's training samples sorted by x0
# (whole rows, so multidimensional units stay intact; the panel itself is only drawn for 1-D x)
sample_x = train_x_last[np.argsort(train_x_last[:, 0])]
cate_true = experiment.true_tau(sample_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)

results_list = []
for tag, _ in learners:
    reps = np.array(bias_reps[tag])  # shape (N_REPS, len(test_scenarios))
    bias_mean = reps.mean(axis=0)  # signed mean bias across reps
    # 95% t-interval for the mean bias: reps are i.i.d. training draws against a fixed test sample
    bias_ci_half = stats.t.ppf(0.975, df=N_REPS - 1) * reps.std(axis=0, ddof=1) / np.sqrt(N_REPS)
    results_list.append({
        "scenario_name": f"{run_tag}_{tag}",
        "label": tag,
        "test_distribution": test_dist.labels,
        "bias": bias_mean,
        "bias_ci_half": bias_ci_half,
        "sample_x": sample_x,
        "cate_true": cate_true,
        "sample_pred": fitted_learners_last[tag].predict(sample_x),
    })
# training data is identical across learners (paired comparison), so only draw the rug once
results_list[0]["train_x"] = train_x_last
results_list[0]["train_a"] = train_a_last

# -----------------------------------------------------------------------------
# 3. Save snapshot + plot to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_experiment

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, SOURCE_SNAPSHOT, output_root=output_root)
plot_experiment(results_list, run_dir / "plot.png")
print(f"Saved to {run_dir}")
