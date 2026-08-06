"""Plot ATE bias vs. test distribution for one or more scenarios.

Each entry of results_list is a dict as built by running_code scenario scripts:
{"scenario_name": str, "test_distribution": list[str], "bias": list[float]}, where each
test_distribution entry is a string describing the test population at that point.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_bias_vs_test_mean(results_list, save_path):
    """One line per scenario in results_list, ATE bias against the test distribution."""
    fig, ax = plt.subplots()

    for results in results_list:
        ax.plot(results["test_distribution"], results["bias"], marker="o", label=results["scenario_name"])

    ax.set_xlabel("Test distribution")
    ax.set_ylabel("ATE bias")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
