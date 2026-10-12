"""Figures for the thesis, sized for direct inclusion (\\includegraphics[width=\\textwidth]): each is
saved as PDF (vector, for the thesis) and PNG (preview).

plot_experiment: synthetic experiments - actual vs. mean estimated ATE (a) and mean squared error (b)
per test distribution, from each results dict's "estimated_ate_reps" (n_reps, n_scenarios) and
"actual_ate" (n_scenarios); "test_distribution" holds the tick labels, "axis_label" the x-axis label. If a results dict carries "ci_half_width_reps" (same shape as the
estimates), panel (a) also shades its mean 95% CI.

plot_coverage: ATE-estimator experiment - 95% CI coverage (a) and mean CI width (b).

plot_ist: IST real-data experiment - transported estimates vs. each target country's randomized estimate.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patches import Patch

TEXT_WIDTH = 15 / 2.54  # inches: thesis \textwidth = 15 cm (ETHDASA.sty, SfS template; 11 pt body, lmodern)
PANEL_HEIGHT = 2.7  # inches, for the two-panel figures
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]  # colourblind-safe
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    # Computer Modern, matching LaTeX's default body font
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,  # cmr10 has no unicode minus
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=OKABE_ITO),
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 300,
})


def _save(fig, save_path):
    """Save as PNG and, next to it, as PDF."""
    save_path = Path(save_path)
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def _two_panels():
    return plt.subplots(1, 2, figsize=(TEXT_WIDTH, PANEL_HEIGHT), layout="constrained")


def _bottom_legend(fig, handles, ncol=None):
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles))


def _format_scenario_axis(ax, results_list):
    """Short tick labels per sweep point ("test_distribution"), the family named on the axis ("axis_label")."""
    labels = results_list[0]["test_distribution"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_xlabel(results_list[0].get("axis_label", "Target distribution"))


def _style(i):
    return dict(color=OKABE_ITO[i % len(OKABE_ITO)], marker=MARKERS[i % len(MARKERS)])


def plot_experiment(results_list, save_path):
    fig, (ax_ate, ax_mse) = _two_panels()
    x_index = np.arange(len(results_list[0]["test_distribution"]))

    # (a) actual ATE and each estimator's mean estimate, with its mean 95% CI if available
    handles = []
    has_ci = False
    for i, results in enumerate(results_list):
        mean_estimate = np.mean(results["estimated_ate_reps"], axis=0)
        handles += ax_ate.plot(x_index, mean_estimate, label=results["label"], **_style(i))
        if "ci_half_width_reps" in results:
            has_ci = True
            half_width = np.mean(results["ci_half_width_reps"], axis=0)
            ax_ate.fill_between(x_index, mean_estimate - half_width, mean_estimate + half_width,
                                color=_style(i)["color"], alpha=0.15, linewidth=0)
    # actual ATE depends only on the test scenario, so it's shared across estimators
    handles += ax_ate.plot(x_index, results_list[0]["actual_ate"], color="black", linestyle="--", marker="x",
                           label="Actual ATE")
    if has_ci:
        handles.append(Patch(color="grey", alpha=0.3, linewidth=0, label="Mean 95% CI"))
    _format_scenario_axis(ax_ate, results_list)
    ax_ate.set_ylabel("ATE (actual and mean estimate)")
    ax_ate.set_title("(a)", loc="left")

    # (b) mean over reps of (estimated ATE - actual ATE)^2, log scale (MSEs span orders of magnitude)
    for i, results in enumerate(results_list):
        errors = np.asarray(results["estimated_ate_reps"]) - np.asarray(results["actual_ate"])
        ax_mse.plot(x_index, np.mean(errors ** 2, axis=0), **_style(i))
    _format_scenario_axis(ax_mse, results_list)
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("MSE of the ATE estimate")
    ax_mse.set_title("(b)", loc="left")

    _bottom_legend(fig, handles, ncol=len(handles) if len(handles) <= 4 else 3)
    _save(fig, save_path)


def plot_coverage(results_list, n_reps, save_path):
    """95% CI coverage (a) and mean CI width (b) per test distribution, from each results dict's
    "coverage" and "mean_ci_width". The grey band is the Monte Carlo range of an exact 95% CI over
    n_reps reps: 0.95 +- 1.96 sqrt(0.95 * 0.05 / n_reps).
    """
    fig, (ax_cov, ax_width) = _two_panels()
    x_index = np.arange(len(results_list[0]["test_distribution"]))

    mc_half_width = 1.96 * np.sqrt(0.95 * 0.05 / n_reps)
    ax_cov.axhspan(0.95 - mc_half_width, 0.95 + mc_half_width, color="grey", alpha=0.2, linewidth=0)
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    handles = []
    for i, results in enumerate(results_list):
        handles += ax_cov.plot(x_index, results["coverage"], label=results["label"], **_style(i))
        ax_width.plot(x_index, results["mean_ci_width"], **_style(i))
    handles.append(Patch(color="grey", alpha=0.35, linewidth=0, label="95% $\\pm$ Monte Carlo error"))

    _format_scenario_axis(ax_cov, results_list)
    ax_cov.set_ylim(0, 1.02)
    ax_cov.set_ylabel("Coverage of the 95% CI")
    ax_cov.set_title("(a)", loc="left")
    _format_scenario_axis(ax_width, results_list)
    ax_width.set_ylim(bottom=0)
    ax_width.set_ylabel("Mean width of the 95% CI")
    ax_width.set_title("(b)", loc="left")

    _bottom_legend(fig, handles)
    _save(fig, save_path)


COUNTRY_NAMES = {"UK": "United Kingdom", "ITAL": "Italy", "SWIT": "Switzerland"}


def plot_ist(rows, learner_labels, save_path):
    """Per target country (one row dict each, values in percentage points): the randomized estimate
    with its 95% CI and each learner's transported estimate (learner labels only; "transported" belongs
    in the caption). The learners' own uncertainty is not shown.
    """
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, PANEL_HEIGHT), layout="constrained")
    x_index = np.arange(len(rows))
    offsets = np.linspace(-0.24, 0.24, len(learner_labels) + 1)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")

    benchmark = np.array([r["benchmark_rd"] for r in rows])
    ci = np.array([[r["benchmark_rd"] - r["benchmark_ci_low"], r["benchmark_ci_high"] - r["benchmark_rd"]] for r in rows]).T
    handles = [ax.errorbar(x_index + offsets[0], benchmark, yerr=ci, fmt="D", color="black", capsize=3,
                           markersize=4, linewidth=1, label="Randomized estimate in target (95% CI)")]
    for i, (offset, label) in enumerate(zip(offsets[1:], learner_labels)):
        handles += ax.plot(x_index + offset, [r[label] for r in rows], linestyle="none", label=label, **_style(i))

    ax.set_xticks(x_index)
    ax.set_xticklabels([COUNTRY_NAMES.get(r["country"], r["country"]) for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Target country")
    ax.set_ylabel("Risk difference (pp)")
    _bottom_legend(fig, handles)
    _save(fig, save_path)
