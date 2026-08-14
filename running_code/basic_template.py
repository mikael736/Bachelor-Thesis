"""Annotated template: explains the experiment pipeline (training/test distributions,
response surface, noise, propensity, learner) for an outsider, then fits a CATE learner and
evaluates its estimated ATE against the true ATE, using base_code/experiment.py's fit_learner
and evaluate helpers. A single train/test sample, no replication - see e.g.
normal_linear_mu0+constant_XLearner.py for the full replicated, swept, and plotted pattern.
Copy this file to start a new scenario script, then strip these explanations down to comments
that describe the scenario itself.
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
# Experiment configuration: pick the training distribution, test/deployment
# distribution, response surface, noise, and propensity for this run. Each
# option lives in its matching base_code script, where more variants can be
# added later. #editable lines are choices to make; #derived lines are
# mechanical and shouldn't need touching.
# =============================================================================
# FILL OUT FOR YOUR EXPERIMENT

# base_code/covariates.py - training-distribution covariates
train_x = covariates.normal(config.TRAIN_POPULATION_SIZE, mean=0.0, sd=1.0, rng=rng)  #editable

# base_code/response_surface.py - mu0's shape, and what gets added on top for mu1 (the treatment effect);
# shared by train and test
mu0_shape = lambda x: response_surface.linear(x, slope=1.0)  #editable
treatment_effect_shape = lambda x: response_surface.constant(x, value=2.0)  #editable

# base_code/noise.py - noise added to the training potential outcomes
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)  #editable

# base_code/propensity.py - treatment assignment on the training sample
propensity_shape = lambda x: propensity.constant(x, p=0.5)  #editable

# base_code/cate_learners.py - the CATE learner to train and evaluate
learner = cate_learners.XLearner(base_learner="rf")  #editable

# base_code/covariates.py - test/deployment-distribution covariates (shifted mean)
test_x = covariates.normal(config.TEST_POPULATION_SIZE, mean=5.0, sd=1.0, rng=rng)  #editable

# =============================================================================

# -----------------------------------------------------------------------------
# Training/Prediction
# -----------------------------------------------------------------------------

# fit the CATE learner on the training sample (simulates outcomes from mu0_shape/mu1_shape,
# noise, and propensity internally - see base_code/experiment.py)
learner, a_train = experiment.fit_learner(
    train_x,
    mu0_shape=mu0_shape,
    treatment_effect_shape=treatment_effect_shape,
    noise_sampler=noise_sampler,
    propensity_shape=propensity_shape,
    learner=learner,
    rng=rng,
)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# true mu0(x)/mu1(x) on the test distribution (no noise, no propensity: we know the truth),
# compared against the fitted learner's predictions
true, estimated, bias = experiment.evaluate(
    learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape
)

print(f"True ATE:      {true:.4f}")
print(f"Estimated ATE: {estimated:.4f}")
print(f"ATE bias:      {bias:.4f}")
