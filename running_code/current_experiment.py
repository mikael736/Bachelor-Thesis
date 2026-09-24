import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0,str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np
from scipy import stats

import cate_learners
import config
import covariates
import experiment
import noise
import propensity
import response_surface
import save_run

seed_seq = np.random.SeedSequence(config.SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(config.N_REPS)]
# fixed target sample, drawn once and reused across all reps
test_rng = np.random.default_rng(config.SEED)

# -----------------------------------------------------------------------------
# 1. Training setup (edit per experiment)
# -----------------------------------------------------------------------------

# training covariate distribution
population_tag = "beta5-5"
train_covariates = lambda rng: covariates.beta(config.TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
mu0_tag = "constant"
mu0_shape = lambda x: response_surface.constant(x, value=100.0)
mu1_tag = "mu0+exponential"
treatment_effect_shape = lambda x: response_surface.exponential(x, multiplier=4)

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_tag = "inv-linear"
propensity_shape = lambda x: propensity.linear(1 - x)

# CATE learners to fit and compare
learners = [
    ("XLearnerRF", lambda rng: cate_learners.XLearner(base_learner="rf", random_state=config.SEED)),
    ("TLearnerRF", lambda rng: cate_learners.TLearner(base_learner="rf", random_state=config.SEED)),
    ("DRLearnerRF", lambda rng: cate_learners.DRLearner(base_learner="rf", random_state=config.SEED)),
]

# test-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep1"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output dir name, composed from the *_tag values above - keep tags current when you edit them
learner_tag = "-vs-".join(tag for tag, _ in learners)
run_tag = f"{mu0_tag}_{mu1_tag}_{propensity_tag}"

# -----------------------------------------------------------------------------
# 2. Evaluate each learner on each test scenario, over config.N_REPS replications. Each
# replication's training data is shared across all learners for a paired comparison - only the
# learner differs, not the data.
# -----------------------------------------------------------------------------

test_scenarios = test_dist.covariates(test_rng)

bias_reps = {tag: [] for tag, _ in learners}
fitted_learners_last, train_x_last, train_a_last = None, None, None
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{config.N_REPS}")
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

# CATE panel: true vs. predicted tau(x), evaluated at sorted training samples from the last rep
sample_x = np.sort(train_x_last, axis=0)
cate_true = experiment.true_tau(sample_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)

results_list = []
for tag, _ in learners:
    reps = np.array(bias_reps[tag])  # shape (config.N_REPS, len(test_scenarios))
    bias_mean = reps.mean(axis=0)  # signed mean bias across reps
    # 95% t-interval for the mean bias: reps are i.i.d. training draws against a fixed test sample
    bias_ci_half = stats.t.ppf(0.975, df=config.N_REPS - 1) * reps.std(axis=0, ddof=1) / np.sqrt(config.N_REPS)
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
