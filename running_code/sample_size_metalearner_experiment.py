"""Meta-learner experiment (metalearner_experiment.py) for growing study sample size n: how do the
MSEs of the T-, X- and DR-learner's plug-in estimates change with n? Same settings, except fewer
replications (running costs grow with n). Per experiment and n, a fresh SeedSequence(SEED).spawn(N_REPS)
as in the main runs, so at n = 1000 the reps reproduce the first N_REPS reps of the main runs (check).
One output directory per experiment; its results are rewritten after every n, so an interrupted run
keeps the finished sample sizes.
"""
import csv
import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import matplotlib.pyplot as plt
import numpy as np

import cate_learners
import covariates
import noise
import propensity
import response_surface
import save_run

# -----------------------------------------------------------------------------
# 0. Run settings - as in metalearner_experiment.py, except N_REPS and the sample sizes
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 20  # fewer than the main runs (100), since the running costs grow with n
TRAIN_POPULATION_SIZES = [1000, 2000, 5000, 10_000]
TEST_POPULATION_SIZE = 100_000  # per test-distribution sweep point, redrawn every rep
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)

# -----------------------------------------------------------------------------
# 1. Training setup, as in metalearner_experiment.py
# -----------------------------------------------------------------------------

# experiments E1-E4, run one after another: mu0(x), tau(x) = mu1(x) - mu0(x) and the jump locations
# of tau (for the exact 1-D integral); [] if continuous
EXPERIMENTS = {
    1: (lambda x: response_surface.sine(x[:, 0], amplitude=3.0, frequency=2.0),
        lambda x: response_surface.constant(x[:, 0], value=1.0), []),
    2: (lambda x: response_surface.linear(x[:, 0], slope=7.0),
        lambda x: response_surface.linear(x[:, 0], slope=7.0), []),
    3: (lambda x: response_surface.linear(x[:, 0], slope=7.0),
        lambda x: response_surface.step(x[:, 0], jump=3.0, threshold=0.7), [0.7]),
    4: (lambda x: response_surface.sine(x[:, 0], amplitude=3.0, frequency=2.0),
        lambda x: response_surface.linear(x[:, 0], slope=7.0), []),
}

noise_sampler = lambda x, rng: noise.homoskedastic_gaussian(x[:, 0], sd=1.0, rng=rng)
propensity_shape = lambda x: propensity.sigmoid(x[:, 0], center=0.5, multiplier=5.0)
test_dist = covariates.TEST_DISTRIBUTIONS["beta-shape-sweep"]

rf = dict(base_learner="rf", regressor_params=REGRESSOR_PARAMS, random_state=SEED)
learners = [
    ("T-learner", lambda: cate_learners.TLearner(**rf)),
    ("X-learner", lambda: cate_learners.XLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
    ("DR-learner", lambda: cate_learners.DRLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
]

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"


def save_results(run_dir, estimates, actual_ate):
    """Summary table, per-rep estimates and a quick-look plot of the MSE against n, for the sample
    sizes finished so far. estimates: {(n, learner): array (N_REPS, n_scenarios)}."""
    sizes = sorted({n for n, _ in estimates})
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "learner", "test_distribution", "actual_ate", "mean_estimate", "bias", "variance",
                         "mse", "mcse_mse"])
        for (n, tag), est in estimates.items():
            squared_errors = (est - actual_ate) ** 2
            for i, dist_label in enumerate(test_dist.labels):
                writer.writerow([n, tag, dist_label, actual_ate[i], est[:, i].mean(), est[:, i].mean() - actual_ate[i],
                                 est[:, i].var(ddof=1), squared_errors[:, i].mean(),
                                 squared_errors[:, i].std(ddof=1) / np.sqrt(N_REPS)])
    np.savez(run_dir / "estimates.npz", actual_ate=actual_ate,
             **{f"n{n}_{tag}": est for (n, tag), est in estimates.items()})

    # quick look: one panel per target distribution, MSE +- 2 MCSE against n
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for i, ax in enumerate(axes.flat[:len(test_dist.labels)]):
        for tag, _ in learners:
            squared_errors = [(estimates[n, tag][:, i] - actual_ate[i]) ** 2 for n in sizes]
            mse = np.array([s.mean() for s in squared_errors])
            mcse = np.array([s.std(ddof=1) / np.sqrt(N_REPS) for s in squared_errors])
            ax.errorbar(sizes, mse, yerr=2 * mcse, marker="o", capsize=3, label=tag)
        ax.set(xscale="log", yscale="log", title=test_dist.labels[i])
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.supxlabel("Study sample size n")
    fig.supylabel("MSE of the plug-in estimate")
    fig.tight_layout()
    fig.savefig(run_dir / "plot.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Per experiment and n: N_REPS replications, each with new training data and new target samples,
# shared across all learners
# -----------------------------------------------------------------------------

for exp_num, (mu0_shape, treatment_effect_shape, treatment_effect_breakpoints) in EXPERIMENTS.items():
    run_dir = save_run.save_run(f"sample_size_metalearner_experiment_{exp_num}", SOURCE_SNAPSHOT, output_root=output_root)
    # exact actual ATE (1-D), as in the main runs
    actual_ate = np.array([
        dist.expect(lambda x: treatment_effect_shape(np.array([[x]]))[0], points=treatment_effect_breakpoints or None)
        for dist in test_dist.distributions
    ])
    estimates = {}
    for n in TRAIN_POPULATION_SIZES:
        rep_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(N_REPS)]
        estimated_ate_reps = {tag: [] for tag, _ in learners}
        for rep_num, rep_rng in enumerate(rep_rngs, start=1):
            print(f"E{exp_num}, n = {n}: round {rep_num}/{N_REPS}")
            x_train = covariates.beta(n, a=5.0, b=5.0, rng=rep_rng)
            mu0_train = mu0_shape(x_train)
            mu1_train = mu0_train + treatment_effect_shape(x_train)
            e0, e1 = noise_sampler(x_train, rep_rng)
            a_train = rep_rng.binomial(n=1, p=propensity_shape(x_train))
            y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)
            test_scenarios = test_dist.covariates(TEST_POPULATION_SIZE, rep_rng)
            for tag, make_learner in learners:
                fitted_learner = make_learner().fit(x_train, y_train, a_train)
                estimated_ate_reps[tag].append([fitted_learner.predict(test_x).mean() for test_x in test_scenarios])
        for tag, _ in learners:
            estimates[n, tag] = np.array(estimated_ate_reps[tag])  # shape (N_REPS, n_scenarios)
        save_results(run_dir, estimates, actual_ate)
        print(f"Saved E{exp_num}, n = {n} to {run_dir}")
