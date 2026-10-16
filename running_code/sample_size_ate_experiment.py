"""ATE-estimator experiment (ate_estimator_experiment.py) for growing study sample size n: how do the
MSE and the CI coverage of the plug-in estimate (naive CI) and the AIPW-corrected estimate change with
n, and does the ratio of estimated to actual variance of the AIPW-corrected estimate approach 1
(Zeng et al., Theorem 5)? Same settings, except fewer replications. The target sample size m grows
with n (m = n), so the ratio m / n stays at 1 as in the main runs.
Per experiment and n, a fresh SeedSequence(SEED).spawn(N_REPS) as in the main runs, so at n = 1000 the
reps reproduce the first N_REPS reps of the main runs (check). One output directory per experiment;
its results are rewritten after every n, so an interrupted run keeps the finished sample sizes.
"""
import csv
import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, as in plotting.py
import matplotlib.pyplot as plt
import numpy as np

import ate_estimators
import covariates
import noise
import propensity
import response_surface
import save_run

# -----------------------------------------------------------------------------
# 0. Run settings - as in ate_estimator_experiment.py, except N_REPS and the sample sizes
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 20  # fewer than the main runs (100), since the running costs grow with n
TRAIN_POPULATION_SIZES = [1000, 2000, 5000, 10_000]  # source sample n; target sample m = n per sweep point
N_FOLDS = 5
MAX_DENSITY_RATIO = 20
PROPENSITY_CLIP = 0.05
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=5)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=5)
DENSITY_RATIO_PARAMS = dict(max_features="sqrt", min_samples_leaf=100)

# -----------------------------------------------------------------------------
# 1. Data-generating process, as in ate_estimator_experiment.py
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

ESTIMATORS = [("Plug-in (naive CI)", "plug_in"), ("AIPW-corrected", "aipw")]  # both T-learner
output_root = Path(__file__).resolve().parent.parent / "visualisation_output"


def save_results(run_dir, estimates, actual_ate):
    """Summary table, per-rep estimates and quick-look plots against n, for the sample sizes finished
    so far. estimates: {n: {key: array (N_REPS, n_scenarios)}}."""
    sizes = sorted(estimates)
    rows = []  # (n, label, i, mse, mcse, coverage, width) for the summary and the plots
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "estimator", "test_distribution", "actual_ate", "mean_estimate", "bias", "variance",
                         "mse", "mcse_mse", "coverage", "mean_ci_width", "est_var", "var_ratio_est_to_actual",
                         "ratio_capped_share"])
        for n in sizes:
            est = estimates[n]
            for label, key in ESTIMATORS:
                point, half_width = est[key], 1.96 * est[f"se_{key}"]
                squared_errors = (point - actual_ate) ** 2
                covered = np.abs(point - actual_ate) <= half_width
                variance = point.var(axis=0, ddof=1)
                est_var = (est[f"se_{key}"] ** 2).mean(axis=0)  # mean over reps of the estimated variance
                for i, dist_label in enumerate(test_dist.labels):
                    row = [n, label, dist_label, actual_ate[i], point[:, i].mean(), point[:, i].mean() - actual_ate[i],
                           variance[i], squared_errors[:, i].mean(),
                           squared_errors[:, i].std(ddof=1) / np.sqrt(N_REPS), covered[:, i].mean(),
                           2 * half_width[:, i].mean(), est_var[i], est_var[i] / variance[i],
                           est["ratio_capped_share"][:, i].mean()]
                    writer.writerow(row)
                    rows.append(dict(n=n, label=label, i=i, mse=row[7], mcse=row[8], coverage=row[9],
                                     var_ratio=row[12]))
    np.savez(run_dir / "estimates.npz", actual_ate=actual_ate,
             **{f"n{n}_{key}": values for n in sizes for key, values in estimates[n].items()})

    # quick look: one panel per target distribution, against n - MSE +- 2 MCSE, coverage, and the
    # ratio of estimated to actual variance
    for name, ylabel, value in [("plot_mse", "MSE", "mse"), ("plot_coverage", "Coverage of the 95% CI", "coverage"),
                                ("plot_var_ratio", "Estimated / actual variance", "var_ratio")]:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=value != "mse")
        for i, ax in enumerate(axes.flat[:len(test_dist.labels)]):
            for label, _ in ESTIMATORS:
                sel = [r for r in rows if r["label"] == label and r["i"] == i]
                y = np.array([r[value] for r in sel])
                yerr = 2 * np.array([r["mcse"] for r in sel]) if value == "mse" else None
                ax.errorbar([r["n"] for r in sel], y, yerr=yerr, marker="o", capsize=3, label=label)
            if value == "coverage":
                ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8)
            if value == "var_ratio":
                ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.set(xscale="log", yscale="log" if value != "coverage" else "linear", title=test_dist.labels[i])
        axes.flat[-1].axis("off")
        axes.flat[0].legend()
        fig.supxlabel("Study sample size n")
        fig.supylabel(ylabel)
        fig.tight_layout()
        fig.savefig(run_dir / f"{name}.png", dpi=150)
        plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Per experiment and n: N_REPS replications, each with new source and target samples, both
# estimators with their CIs per sweep point
# -----------------------------------------------------------------------------

for exp_num, (mu0_shape, treatment_effect_shape, treatment_effect_breakpoints) in EXPERIMENTS.items():
    run_dir = save_run.save_run(f"sample_size_ate_experiment_{exp_num}", SOURCE_SNAPSHOT, output_root=output_root)
    # exact actual ATE (1-D), as in the main runs
    actual_ate = np.array([
        dist.expect(lambda x: treatment_effect_shape(np.array([[x]]))[0], points=treatment_effect_breakpoints or None)
        for dist in test_dist.distributions
    ])
    estimates = {}
    for n in TRAIN_POPULATION_SIZES:
        rep_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(N_REPS)]
        reps = {key: [] for key in ("plug_in", "se_plug_in", "aipw", "se_aipw", "ratio_capped_share")}
        for rep_num, rep_rng in enumerate(rep_rngs, start=1):
            print(f"E{exp_num}, n = {n}: round {rep_num}/{N_REPS}")
            x_train = covariates.beta(n, a=5.0, b=5.0, rng=rep_rng)
            mu0_train = mu0_shape(x_train)
            mu1_train = mu0_train + treatment_effect_shape(x_train)
            e0, e1 = noise_sampler(x_train, rep_rng)
            a_train = rep_rng.binomial(n=1, p=propensity_shape(x_train))
            y_train = np.where(a_train == 1, mu1_train + e1, mu0_train + e0)
            x_targets = test_dist.covariates(n, rep_rng)  # m = n

            rep_results = ate_estimators.transport_t_learner(
                x_train, y_train, a_train, x_targets,
                n_folds=N_FOLDS, propensity_clip=PROPENSITY_CLIP, max_density_ratio=MAX_DENSITY_RATIO,
                regressor_params=REGRESSOR_PARAMS, classifier_params=CLASSIFIER_PARAMS,
                density_ratio_params=DENSITY_RATIO_PARAMS, random_state=SEED,
            )
            for key in reps:
                reps[key].append([result[key] for result in rep_results])
        estimates[n] = {key: np.array(values) for key, values in reps.items()}  # each (N_REPS, n_scenarios)
        save_results(run_dir, estimates, actual_ate)
        print(f"Saved E{exp_num}, n = {n} to {run_dir}")
