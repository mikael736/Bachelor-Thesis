"""Plot the CATE fit (left panel) and ATE bias vs. test distribution (right panel)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _plot_cate(ax, results_list):
    first = results_list[0]
    ax.plot(np.asarray(first["sample_x"]).ravel(), np.asarray(first["cate_true"]).ravel(), color="black", label="True CATE")

    for results in results_list:
        label = results.get("label", "Predicted CATE")
        ax.plot(np.asarray(results["sample_x"]).ravel(), np.asarray(results["sample_pred"]).ravel(), label=label)

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\tau(x)$")
    ax.legend()


def _plot_ate(ax, results_list):
    """Bottom axis: ATE bias per test scenario, equally spaced. Top axis (separate scale):
    red/blue rug ticks for train_x by train_a (treated/control).
    """
    handles = []
    first = results_list[0]
    n_scenarios = len(first["test_distribution"])
    x_index = np.arange(n_scenarios)
    for results in results_list:
        bias = np.asarray(results["bias"])
        label = results.get("label", "ATE bias")
        line, = ax.plot(x_index, bias, marker="o", label=label)
        handles.append(line)

    ax.set_xticks(x_index)
    ax.set_xticklabels(first["test_distribution"])
    ax.set_xlim(-0.5, n_scenarios - 0.5)
    ax.set_xlabel("Test distribution")
    ax.set_ylabel("ATE bias")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    train_carrier = next((r for r in results_list if r.get("train_x") is not None), None)
    if train_carrier is None:
        raise ValueError("_plot_ate requires one scenario with train_x/train_a (training data is shared)")
    if train_carrier.get("train_a") is None:
        raise ValueError("_plot_ate requires train_a alongside train_x")

    train_x = np.asarray(train_carrier["train_x"]).ravel()
    train_a = np.asarray(train_carrier["train_a"]).ravel()
    ax_train = ax.twiny()
    ax_train.vlines(
        train_x[train_a == 1], 0.97, 1.0, transform=ax_train.get_xaxis_transform(),
        color="red", alpha=0.15, linewidth=0.8,
    )
    ax_train.vlines(
        train_x[train_a == 0], 0.97, 1.0, transform=ax_train.get_xaxis_transform(),
        color="blue", alpha=0.15, linewidth=0.8,
    )
    ax_train.set_xlabel("Train $x$")

    def _rug_handle(color, label):
        return Line2D(
            [0], [0], color=color, alpha=0.5, marker="|", linestyle="None",
            markersize=10, markeredgewidth=1.5, label=label,
        )

    handles.append(_rug_handle("red", "Train $x$ (treated)"))
    handles.append(_rug_handle("blue", "Train $x$ (control)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


def plot_experiment(results_list, save_path):
    fig, (ax_cate, ax_ate) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_cate(ax_cate, results_list)
    _plot_ate(ax_ate, results_list)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
