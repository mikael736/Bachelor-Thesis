"""plot_experiment: synthetic experiments - actual vs. estimated ATE (left) and mean squared ATE
error (right) per test distribution. Both panels are derived from each results dict's
"estimated_ate_reps" (n_reps, n_scenarios) and "actual_ate" (n_scenarios), so overwriting
"actual_ate" (e.g. with exact values) updates both.

plot_ist: IST real-data experiment - transported estimates vs. each target country's randomized estimate.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _format_scenario_axis(ax, results_list):
    labels = results_list[0]["test_distribution"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("Test distribution")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _plot_ate(ax, results_list):
    """Actual ATE and each learner's mean estimated ATE across reps. Returns the legend handles."""
    x_index = np.arange(len(results_list[0]["test_distribution"]))
    handles = [
        ax.plot(x_index, np.mean(results["estimated_ate_reps"], axis=0), marker="o", label=results["label"])[0]
        for results in results_list
    ]
    # actual ATE depends only on the test scenario, so it's shared across learners
    handles += ax.plot(
        x_index, results_list[0]["actual_ate"], color="black", linestyle="--", marker="x", label="Actual ATE",
    )

    _format_scenario_axis(ax, results_list)
    ax.set_ylabel("ATE (estimated: mean across reps)")
    return handles


def _plot_squared_error(ax, results_list):
    """mean over reps of (estimated ATE - actual ATE)^2, per scenario."""
    x_index = np.arange(len(results_list[0]["test_distribution"]))
    for results in results_list:
        errors = np.asarray(results["estimated_ate_reps"]) - np.asarray(results["actual_ate"])
        ax.plot(x_index, np.mean(errors ** 2, axis=0), marker="o", label=results["label"])

    _format_scenario_axis(ax, results_list)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean squared ATE error")


def plot_experiment(results_list, save_path):
    fig, (ax_ate, ax_sq_err) = plt.subplots(1, 2, figsize=(14, 6))
    handles = _plot_ate(ax_ate, results_list)
    _plot_squared_error(ax_sq_err, results_list)

    # one legend below both panels (learner colours match across panels)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(handles), frameon=False)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_ist(rows, learner_labels, save_path):
    """Per target country (one row dict each, values in percentage points): the randomized estimate
    with its 95% CI and each learner's transported estimate. The learners' own uncertainty is not shown.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x_index = np.arange(len(rows))
    offsets = np.linspace(-0.24, 0.24, len(learner_labels) + 1)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")

    benchmark = np.array([r["benchmark_rd"] for r in rows])
    ci = np.array([[r["benchmark_rd"] - r["benchmark_ci_low"], r["benchmark_ci_high"] - r["benchmark_rd"]] for r in rows]).T
    ax.errorbar(
        x_index + offsets[0], benchmark, yerr=ci, fmt="s", color="black", capsize=4,
        label="Randomized estimate in target (95% CI)",
    )
    for offset, label in zip(offsets[1:], learner_labels):
        ax.plot(x_index + offset, [r[label] for r in rows], "o", label=f"{label} (transported)")

    ax.set_xticks(x_index)
    ax.set_xticklabels([r["country"] for r in rows])
    ax.set_xlabel("Target country")
    ax.set_ylabel("Aspirin effect: risk difference (percentage points)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
