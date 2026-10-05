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
N_REPS = 50  # training/evaluation replications per learner and test scenario
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
mu0_tag = "linear"
mu0_shape = lambda x: response_surface.linear(x[:, 0], slope=50.0)
mu1_tag = "mu0+linear"
treatment_effect_shape = lambda x: response_surface.linear(x[:, 0], slope=50.0)

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_tag = "sigmoid"
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)

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
run_tag = f"{mu0_tag}_{mu1_tag}_{propensity_tag}"

# -----------------------------------------------------------------------------
# 2. Evaluate each learner on each test scenario, over N_REPS replications. Each
# replication's training data is shared across all learners for a paired comparison - only the
# learner differs, not the data.
# -----------------------------------------------------------------------------

test_scenarios = test_dist.covariates(TEST_POPULATION_SIZE, test_rng)

estimated_ate_reps = {tag: [] for tag, _ in learners}
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{N_REPS}")
    train_x = train_covariates(rep_rng)
    mu0_train = mu0_shape(train_x)
    mu1_train = mu0_train + treatment_effect_shape(train_x)
    e0, e1 = noise_sampler(train_x, rep_rng)
    a_train = rep_rng.binomial(n=1, p=propensity_shape(train_x))
    y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)

    for tag, make_learner in learners:
        fitted_learner = make_learner(rep_rng).fit(train_x, y_train, a_train)
        estimated_ate_reps[tag].append([fitted_learner.predict(test_x).mean() for test_x in test_scenarios])

# actual ATE per scenario, approximated on the fixed test sample - overwrite with exact values when available
actual_ate = np.array([treatment_effect_shape(test_x).mean() for test_x in test_scenarios])

results_list = [
    {
        "scenario_name": f"{run_tag}_{tag}",
        "label": tag,
        "test_distribution": test_dist.labels,
        "estimated_ate_reps": np.array(estimated_ate_reps[tag]),  # shape (N_REPS, len(test_scenarios))
        "actual_ate": actual_ate,
    }
    for tag, _ in learners
]

# -----------------------------------------------------------------------------
# 3. Save snapshot + plot to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_experiment

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, SOURCE_SNAPSHOT, output_root=output_root)
plot_experiment(results_list, run_dir / "plot.png")
print(f"Saved to {run_dir}")
