"""Plot ATE bias vs. test distribution for one or more scenarios.

Each entry of results_list is a dict as built by running_code scenario scripts:
{"scenario_name": str, "test_distribution": list[str], "bias_mean": array-like, "bias_std":
array-like}, where each test_distribution entry is a string describing the test population at
that point, and bias_mean/bias_std are the mean and standard deviation of the ATE bias across
replications at that point.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_bias_vs_test_mean(results_list, save_path):
    """One line per scenario in results_list: mean ATE bias against the test distribution,
    shaded with a +/-1 SD band across replications.
    """
    fig, ax = plt.subplots()

    for results in results_list:
        mean = np.asarray(results["bias_mean"])
        std = np.asarray(results["bias_std"])
        line, = ax.plot(results["test_distribution"], mean, marker="o", label=results["scenario_name"])
        ax.fill_between(results["test_distribution"], mean - std, mean + std, color=line.get_color(), alpha=0.2)

    ax.set_xlabel("Test distribution")
    ax.set_ylabel("ATE bias")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
