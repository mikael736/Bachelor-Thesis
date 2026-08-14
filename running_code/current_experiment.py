"""Current experiment workbench: hand-edit the training population, response surface, noise,
propensity, and learner(s) below for each new experiment, pick a named test-distribution sweep
from covariates.TEST_DISTRIBUTIONS, then run this file. Every run writes its plot and an exact
snapshot of this file's own source into visualisation_output/<run_tag>/ (see save_run.py), so
each experiment is fully reproducible from what's saved there: re-running an unchanged snapshot
reproduces the same output (the pipeline is deterministic given config.SEED), while changing this
file and reusing the same run_tag gets auto-suffixed into a new directory instead of silently
overwriting a different run.

Supports comparing several learner configs (e.g. a fully flexible baseline vs. one with an
informed mu0) side by side in one plot - each replication simulates its training data once, then
every learner in `learners` is fit on that exact same data, so differences in the resulting bias
curves come only from the learner, not from resampled data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np

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

# -----------------------------------------------------------------------------
# 1. Training: population, treatment mechanism, and CATE learner(s). Refit per replication.
# Hand-edit this section for each new experiment.
# -----------------------------------------------------------------------------

# X ~ Normal(mean=0, sd=1)
population_tag = "normal0-1"
train_covariates = lambda rng: covariates.normal(config.TRAIN_POPULATION_SIZE, mean=0.0, sd=1.0, rng=rng)

# mu0(x) = exp(x)  ->  exponential baseline
mu0_tag = "exponential"
mu0_shape = lambda x: response_surface.exponential(x, scale=1.0)
# mu1(x) = mu0(x) + x  ->  linear treatment effect, tau(x) = x
mu1_tag = "mu0+linear"
treatment_effect_shape = lambda x: response_surface.linear(x, slope=1.0)

# homoskedastic Gaussian noise (sd=1) on both arms
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# e(x) = 0.5: randomized, balanced treatment assignment
propensity_shape = lambda x: propensity.constant(x, p=0.5)

# X-learner comparison: a fully flexible NN baseline vs. one where mu0 (the control/
# natural-history surface) uses a correctly-specified linear model instead, isolating how much an
# informed mu0 alone reduces extrapolation bias while mu1/tau0/tau1 stay flexible (NN) in both.
# random_state must be an int (sklearn doesn't accept a Generator), so draw one off rng.
learners = [
    ("XLearnerNN", lambda rng: cate_learners.XLearner(base_learner="nn", random_state=int(rng.integers(0, 2**32 - 1)))),
    # ("XLearnerNN+mu0Exponential", lambda rng: cate_learners.XLearner(
    #         base_learner="nn", mu0_learner="exponential", random_state=int(rng.integers(0, 2**32 - 1))
    #     ),
    # ),
]

# named test-distribution sweep to evaluate against, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "normal-mean-sweep1"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# identifies this run's output directory - composed from the *_tag values above, so it stays in
# sync with them automatically. Keep each tag current when you change its value above (fold in
# any numeric value you're deliberately comparing, e.g. mu1_tag = "mu0+constant-tau10"); a
# same-tag collision with different code still auto-suffixes rather than silently overwriting
# (see save_run.py), so a forgotten tag update is caught, not lost.
learner_tag = "-vs-".join(tag for tag, _ in learners)
run_tag = f"{population_tag}_{mu0_tag}_{mu1_tag}_{learner_tag}_{test_dist_name}"

# -----------------------------------------------------------------------------
# 2. Evaluate each fitted learner on each test scenario, over config.N_REPS replications.
# Each replication's training data (and test draws) is shared across all learners for a paired
# comparison - only the learner differs, not the data.
# -----------------------------------------------------------------------------

bias_reps = {tag: [] for tag, _ in learners}
train_x_last, train_a_last = None, None
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
    test_scenarios = test_dist.covariates(rep_rng)

    for tag, make_learner in learners:
        fitted_learner = make_learner(rep_rng).fit(train_x, y_train, a_train)
        bias_reps[tag].append([
            experiment.evaluate(fitted_learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
            for test_x in test_scenarios
        ])

    train_x_last, train_a_last = train_x, a_train

results_list = []
for tag, _ in learners:
    reps = np.array(bias_reps[tag])  # shape (config.N_REPS, len(test_dist.positions))
    results_list.append({
        "scenario_name": f"{run_tag}_{tag}",
        "label": tag,
        "test_distribution": test_dist.labels,
        "test_positions": test_dist.positions,
        "bias_mean": reps.mean(axis=0),
        "bias_std": reps.std(axis=0),
    })
# training data is identical across learners (paired comparison), so only draw the rug once
results_list[0]["train_x"] = train_x_last
results_list[0]["train_a"] = train_a_last

# -----------------------------------------------------------------------------
# 3. Save: snapshot this file + the plot into visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plot_bias_vs_test_distribution import plot_bias_vs_test_mean

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, Path(__file__).read_text(encoding="utf-8"), output_root=output_root)
plot_bias_vs_test_mean(results_list, run_dir / "plot.png")
print(f"Saved to {run_dir}")
