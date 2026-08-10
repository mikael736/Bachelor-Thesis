"""Scenario: training population X ~ Normal(0, 1); mu0(x) = 5*1(x>0.5) (step); mu1(x) = mu0(x) + 8*1(x>0.1)
(step treatment effect, tau(x) = 8*1(x>0.1)); fitted with the X-learner (random-forest
base learners) under a neutral, balanced propensity (e(x) = 0.5). Sweeps the test
population's mean from -5 to 5 and records the resulting ATE bias at each point, averaged
over config.N_REPS independent replications (sample, fit, evaluate) with a +/-1 SD band.
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

seed_seq = np.random.SeedSequence(config.SEED)
rep_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(config.N_REPS)]

# -----------------------------------------------------------------------------
# 1. Training: population, treatment mechanism, and CATE learner. Refit per replication.
# -----------------------------------------------------------------------------

# X ~ Normal(mean=0, sd=1)
train_covariates = lambda rng: covariates.normal(config.TRAIN_POPULATION_SIZE, mean=0.0, sd=1.0, rng=rng)

# mu0(x) = 5 * 1(x > 0.5)
mu0_shape = lambda x: response_surface.step(x, jump=5.0, threshold=0.5)
# mu1(x) = mu0(x) + 8 * 1(x > 0.1)  ->  step treatment effect, tau(x) = 8*1(x>0.1)
treatment_effect_shape = lambda x: response_surface.step(x, jump=8.0, threshold=0.1)

# homoskedastic Gaussian noise (sd=1) on both arms
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# e(x) = 0.5: randomized, balanced treatment assignment
propensity_shape = lambda x: propensity.constant(x, p=0.5)

# X-learner, random-forest base learners. random_state must be an int (sklearn doesn't
# accept a Generator), so draw one off rng.
learner = lambda rng: cate_learners.XLearner(base_learner="rf", random_state=int(rng.integers(0, 2**32 - 1)))

# tags describing this scenario - keep in sync with the file name and with each other:
# mu1_tag records the *relationship* to mu0 ("mu0+step" for an additive
# treatment_effect_shape, or the shape's own name if mu1_shape is used independently)
population_tag = "normal"
mu0_tag = "step"
mu1_tag = "mu0+step"
learner_tag = "XLearner"

scenario_name = f"{population_tag}_{mu0_tag}_{mu1_tag}_{learner_tag}"

# -----------------------------------------------------------------------------
# 2. Test scenarios: sweep the test population's mean from -5 to 5, training fixed
# -----------------------------------------------------------------------------

test_means = np.linspace(-5.0, 5.0, 11)
test_distribution = [f"N({mean:.1f},1.0)" for mean in test_means]
test_covariates = lambda mean, rng: covariates.normal(config.TEST_POPULATION_SIZE, mean=mean, sd=1.0, rng=rng)

# -----------------------------------------------------------------------------
# 3. Evaluate the fitted learner on each test scenario, over config.N_REPS replications
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

    test_scenarios = [test_covariates(mean, rep_rng) for mean in test_means]
    bias_reps.append([
        experiment.evaluate(fitted_learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
        for test_x in test_scenarios
    ])

bias_reps = np.array(bias_reps)  # shape (config.N_REPS, len(test_means))

results = {
    "scenario_name": scenario_name,
    "test_distribution": test_distribution,
    "bias_mean": bias_reps.mean(axis=0),
    "bias_std": bias_reps.std(axis=0),
}

# -----------------------------------------------------------------------------
# 4. Plot results
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plot_bias_vs_test_distribution import plot_bias_vs_test_mean

output_dir = Path(__file__).resolve().parent.parent / "visualisation_output"
plot_bias_vs_test_mean([results], output_dir / f"{Path(__file__).stem}.png")
