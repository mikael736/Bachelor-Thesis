"""Entry point: configure an experiment (training/test distributions, response surface,
noise, propensity, learner), fit a CATE learner on the training sample, predict on the
test distribution, and evaluate the estimated ATE against the true ATE.
"""
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np

import config
import covariates
import evaluation
import noise
import propensity
import response_surface
import cate_learners

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
train_x = covariates.normal(mean=0.0, sd=1.0, n=1000, rng=rng)  #editable

# base_code/response_surface.py - the mechanism, shared by train and test
surface = functools.partial(response_surface.linear_homogeneous, intercept=0.0, slope=1.0, treatment_effect=2.0)  #editable

# apply the configured surface to the training covariates
mu0_train, mu1_train = surface(train_x)  #derived

# base_code/noise.py - noise added to the training potential outcomes
e0, e1 = noise.homoskedastic_gaussian(train_x, sd=1.0, rng=rng)  #editable

# add the configured noise to mu0_train/mu1_train
y0_train, y1_train = noise.apply_noise(mu0_train, mu1_train, e0, e1)  #derived

# base_code/propensity.py - treatment assignment on the training sample
e_x = propensity.constant(train_x, p=0.5)  #editable
a_train = propensity.get_assignment(e_x, rng=rng)  #derived
y_train = np.where(a_train == 1, y1_train, y0_train)  #derived

# base_code/covariates.py - test/deployment-distribution covariates (shifted mean)
test_x = covariates.normal(mean=100.0, sd=1.0, n=1000, rng=rng)  #editable

# true mu0(x), mu1(x) on the test distribution, same surface (no noise, no propensity: we know the truth)
mu0_test, mu1_test = surface(test_x)  #derived
tau_test = mu1_test - mu0_test  #derived

# base_code/cate_learners.py - the CATE learner to train and evaluate
learner = cate_learners.TLearner(base_learner="rf")  #editable

# =============================================================================

# -----------------------------------------------------------------------------
# Training/Prediction
# -----------------------------------------------------------------------------

# fit the CATE learner on the training sample, then predict on the test distribution
learner.fit(train_x, y_train, a_train)
tau_hat_test = learner.predict(test_x)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

true = evaluation.true_ate(tau_test)
estimated = evaluation.estimated_ate(tau_hat_test)
bias = evaluation.ate_bias(true, estimated)

print(f"True ATE:      {true:.4f}")
print(f"Estimated ATE: {estimated:.4f}")
print(f"ATE bias:      {bias:.4f}")
