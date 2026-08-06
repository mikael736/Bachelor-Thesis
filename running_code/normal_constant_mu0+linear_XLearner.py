"""Scenario: training population X ~ Normal(0, 1); mu0(x) = 0 (constant); mu1(x) = mu0(x) + 2*x
(linear treatment effect, tau(x) = 2*x); fitted with the X-learner (random-forest
base learners) under a neutral, balanced propensity (e(x) = 0.5). Sweeps the test
population's mean from -5 to 5 and records the resulting ATE bias at each point.
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

rng = np.random.default_rng(config.SEED)

# -----------------------------------------------------------------------------
# 1. Training: population, treatment mechanism, and CATE learner. Fit once.
# -----------------------------------------------------------------------------

# X ~ Normal(mean=0, sd=1)
train_x = covariates.normal(mean=0.0, sd=1.0, n=1000, rng=rng)

# mu0(x) = 0
mu0_shape = lambda x: response_surface.constant(x, value=0.0)
# mu1(x) = mu0(x) + 2*x  ->  linear treatment effect, tau(x) = 2*x
treatment_effect_shape = lambda x: response_surface.linear(x, slope=2.0)

# homoskedastic Gaussian noise (sd=1) on both arms
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# e(x) = 0.5: randomized, balanced treatment assignment
propensity_shape = lambda x: propensity.constant(x, p=0.5)

# X-learner, random-forest base learners
learner = cate_learners.XLearner(base_learner="rf")

# tags describing this scenario - keep in sync with the file name and with each other:
# mu1_tag records the *relationship* to mu0 ("mu0+linear" for an additive
# treatment_effect_shape, or the shape's own name if mu1_shape is used independently)
population_tag = "normal"
mu0_tag = "constant"
mu1_tag = "mu0+linear"
learner_tag = "XLearner"

scenario_name = f"{population_tag}_{mu0_tag}_{mu1_tag}_{learner_tag}"

learner = experiment.fit_learner(
    train_x,
    mu0_shape=mu0_shape,
    treatment_effect_shape=treatment_effect_shape,
    noise_sampler=noise_sampler,
    propensity_shape=propensity_shape,
    learner=learner,
    rng=rng,
)

# -----------------------------------------------------------------------------
# 2. Test scenarios: sweep the test population's mean from -5 to 5, training fixed
# -----------------------------------------------------------------------------

test_means = np.linspace(-5.0, 5.0, 11)
test_scenarios = [covariates.normal(mean=mean, sd=1.0, n=1000, rng=rng) for mean in test_means]
test_distribution = [f"N({mean:.1f},1.0)" for mean in test_means]

# -----------------------------------------------------------------------------
# 3. Evaluate the fitted learner on each test scenario
# -----------------------------------------------------------------------------

biases = [
    experiment.evaluate(learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
    for test_x in test_scenarios
]

results = {
    "scenario_name": scenario_name,
    "test_distribution": test_distribution,
    "bias": biases,
}

# -----------------------------------------------------------------------------
# 4. Plot results
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plot_bias_vs_test_distribution import plot_bias_vs_test_mean

output_dir = Path(__file__).resolve().parent.parent / "visualisation_output"
plot_bias_vs_test_mean([results], output_dir / f"{Path(__file__).stem}.png")
