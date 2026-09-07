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

rng = np.random.default_rng(config.SEED)

# -----------------------------------------------------------------------------
# 1. Training setup (edit per experiment)
# -----------------------------------------------------------------------------

# training covariate distribution
population_tag = "beta5-5"
train_covariates = lambda rng: covariates.beta(config.TRAIN_POPULATION_SIZE, a=5.0, b=5.0, rng=rng)

# response surfaces: mu0(x) and tau(x) = mu1(x) - mu0(x)
mu0_tag = "constant"
mu0_shape = lambda x: response_surface.constant(x, value=0.0)
mu1_tag = "mu0+linear"
treatment_effect_shape = lambda x: response_surface.linear(x, slope=1.0)

# outcome noise
noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x, sd=1.0, rng=rng)

# treatment assignment mechanism
propensity_shape = lambda x: propensity.linear(x)

# CATE learners to fit and compare
learners = [
    ("XLearnerLinear", lambda rng: cate_learners.XLearner(base_learner="linear", random_state=config.SEED)),
    ("TLearnerNN", lambda rng: cate_learners.TLearner(base_learner="nn", random_state=config.SEED)),
]

# test-distribution sweep, from covariates.TEST_DISTRIBUTIONS
test_dist_name = "beta-shape-sweep1"
test_dist = covariates.TEST_DISTRIBUTIONS[test_dist_name]

# output dir name, composed from the *_tag values above - keep tags current when you edit them
learner_tag = "-vs-".join(tag for tag, _ in learners)
run_tag = f"{population_tag}_{mu0_tag}_{mu1_tag}_{learner_tag}_{test_dist_name}"

# -----------------------------------------------------------------------------
# 2. Evaluate each learner on each test scenario (single training draw, shared across learners)
# -----------------------------------------------------------------------------

test_scenarios = test_dist.covariates(rng)

train_x = train_covariates(rng)
y_train, a_train = experiment.simulate_training_outcomes(
    train_x,
    mu0_shape=mu0_shape,
    treatment_effect_shape=treatment_effect_shape,
    noise_sampler=noise_sampler,
    propensity_shape=propensity_shape,
    rng=rng,
)

# CATE panel: true vs. predicted tau(x), evaluated at sorted training samples
sample_x = np.sort(train_x, axis=0)
cate_true = experiment.true_tau(sample_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)

results_list = []
for tag, make_learner in learners:
    fitted_learner = make_learner(rng).fit(train_x, y_train, a_train)
    bias = np.array([
        experiment.evaluate(fitted_learner, test_x, mu0_shape=mu0_shape, treatment_effect_shape=treatment_effect_shape)[2]
        for test_x in test_scenarios
    ])
    results_list.append({
        "scenario_name": f"{run_tag}_{tag}",
        "label": tag,
        "test_distribution": test_dist.labels,
        "bias": bias,
        "sample_x": sample_x,
        "cate_true": cate_true,
        "sample_pred": fitted_learner.predict(sample_x),
    })
# same training data across learners, so only draw the rug once
results_list[0]["train_x"] = train_x
results_list[0]["train_a"] = a_train

# -----------------------------------------------------------------------------
# 3. Save snapshot + plot to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_experiment

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, Path(__file__).read_text(encoding="utf-8"), output_root=output_root)
plot_experiment(results_list, run_dir / "plot.png")
print(f"Saved to {run_dir}")
