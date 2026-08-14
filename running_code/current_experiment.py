"""Current experiment workbench: hand-edit the training population, response surface, noise,
propensity, and learner below for each new experiment, pick a named test-distribution sweep from
covariates.TEST_DISTRIBUTIONS, then run this file. Every run writes its plot and an exact
snapshot of this file's own source into visualisation_output/<run_tag>/ (see save_run.py), so
each experiment is fully reproducible from what's saved there: re-running an unchanged snapshot
reproduces the same output (the pipeline is deterministic given config.SEED), while changing this
file and reusing the same run_tag gets auto-suffixed into a new directory instead of silently
overwriting a different run.
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
# 1. Training: population, treatment mechanism, and CATE learner. Refit per replication.
# Hand-edit this section for each new experiment.
# -----------------------------------------------------------------------------

# X ~ Beta(a=5, b=5): symmetric, mean 0.5
population_tag = "beta5-5"
train_covariates = lambda rng: covariates.beta(config.TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# mu0(x) = x
mu0_tag = "linear"
mu0_shape = lambda x: response_surface.linear(x, slope=1.0)
# mu1(x) = mu0(x) + 2  ->  constant treatment effect, tau(x) = 2
mu1_tag = "mu0+constant-tau"
treatment_effect_shape = lambda x: response_surface.constant(x, value=2.0)

# homoskedastic Gaussian noise (sd=1) on both arms
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# e(x) = 0.5: randomized, balanced treatment assignment
propensity_shape = lambda x: propensity.constant(x, p=0.5)

# X-learner, random-forest base learners. random_state must be an int (sklearn doesn't
# accept a Generator), so draw one off rng.
learner_tag = "XLearnerRF"
learner = lambda rng: cate_learners.XLearner(base_learner="rf", random_state=int(rng.integers(0, 2**32 - 1)))

# named test-distribution sweep to evaluate against, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta_shape_sweep1"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# identifies this run's output directory - composed from the *_tag values above, so it stays in
# sync with them automatically. Keep each tag current when you change its value above (fold in
# any numeric value you're deliberately comparing, e.g. mu1_tag = "mu0+constant-tau10"); a
# same-tag collision with different code still auto-suffixes rather than silently overwriting
# (see save_run.py), so a forgotten tag update is caught, not lost.
run_tag = f"{population_tag}_{mu0_tag}_{mu1_tag}_{learner_tag}_{test_dist_name}"

# -----------------------------------------------------------------------------
# 2. Evaluate the fitted learner on each test scenario, over config.N_REPS replications
# -----------------------------------------------------------------------------

bias_reps = []
for rep_num, rep_rng in enumerate(rep_rngs, start=1):
    print(f"Round {rep_num}/{config.N_REPS}")
    train_x = train_covariates(rep_rng)
    fitted_learner = experiment.fit_learner(
        train_x,
        mu0_shape=mu0_shape,
        treatment_effect_shape=treatment_effect_shape,
        noise_sampler=noise_sampler,
        propensity_shape=propensity_shape,
        learner=learner(rep_rng),
        rng=rep_rng,
    )

    test_scenarios = test_dist.covariates(rep_rng)
    bias_reps.append([
        experiment.evaluate(fitted_learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
        for test_x in test_scenarios
    ])

bias_reps = np.array(bias_reps)  # shape (config.N_REPS, len(test_dist.positions))

results = {
    "scenario_name": run_tag,
    "test_distribution": test_dist.labels,
    "test_positions": test_dist.positions,
    "bias_mean": bias_reps.mean(axis=0),
    "bias_std": bias_reps.std(axis=0),
    "train_x": train_x,  # last replication's training sample, for the rug plot
}

# -----------------------------------------------------------------------------
# 3. Save: snapshot this file + the plot into visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plot_bias_vs_test_distribution import plot_bias_vs_test_mean

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, Path(__file__).read_text(encoding="utf-8"), output_root=output_root)
plot_bias_vs_test_mean([results], run_dir / "plot.png")
print(f"Saved to {run_dir}")
