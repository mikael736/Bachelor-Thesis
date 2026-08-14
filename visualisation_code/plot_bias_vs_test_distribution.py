"""Plot ATE bias vs. test distribution for one or more scenarios.

Each entry of results_list is a dict as built by running_code scenario scripts:
{"scenario_name": str, "test_distribution": list[str], "test_positions": array-like, "bias_mean":
array-like, "bias_std": array-like, "train_x": array-like (optional)}, where each
test_distribution entry is a string describing the test population at that point, test_positions
is that point's numeric x-axis position (e.g. the test population's mean, when the sweep is
mean-anchored - not guaranteed in general), bias_mean/bias_std are the mean and standard
deviation of the ATE bias across replications at that point, and train_x (if given) is a sample
of the covariates the learner was trained on, drawn as a rug of grey ticks so the trained region
- and where the sweep starts extrapolating beyond it - is visible.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_bias_vs_test_mean(results_list, save_path):
    """One line per scenario in results_list: mean ATE bias against the test distribution's
    mean, shaded with a +/-1 SD band across replications. If a scenario carries "train_x", its
    training covariates are drawn as short grey ticks just above the x-axis.
    """
    fig, ax = plt.subplots()

    for results in results_list:
        test_positions = np.asarray(results["test_positions"])
        mean = np.asarray(results["bias_mean"])
        std = np.asarray(results["bias_std"])
        line, = ax.plot(test_positions, mean, marker="o", label="Mean ATE bias")
        ax.fill_between(test_positions, mean - std, mean + std, color=line.get_color(), alpha=0.2)

        train_x = results.get("train_x")
        if train_x is not None:
            ax.vlines(
                train_x, 0.0, 0.03, transform=ax.get_xaxis_transform(),
                color="grey", alpha=0.15, linewidth=0.8,
            )

    first = results_list[0]
    ax.set_xticks(np.asarray(first["test_positions"]))
    ax.set_xticklabels(first["test_distribution"])

    ax.set_xlabel("Test distribution")
    ax.set_ylabel("ATE bias")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    handles, _ = ax.get_legend_handles_labels()  # one per scenario line
    handles.append(Patch(facecolor=line.get_color(), alpha=0.2, label="±1 SD across replications"))
    if any(results.get("train_x") is not None for results in results_list):
        handles.append(Line2D(
            [0], [0], color="grey", alpha=0.5, marker="|", linestyle="None",
            markersize=10, markeredgewidth=1.5, label="Train $x$ samples",
        ))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
