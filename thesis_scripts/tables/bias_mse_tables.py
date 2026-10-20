"""Tables for Appendix "Bias and MSE with Monte Carlo standard errors": bias and MSE of the ATE estimates,
each with its Monte Carlo standard error (MCSE), for every target distribution Beta(alpha, 10 - alpha) of
each experiment E1-E4,
- meta_bias.tex, meta_mse.tex: plug-in ATE estimate of the T-, X- and DR-learner (metalearner_experiment_i),
- ate_bias.tex, ate_mse.tex: plug-in and AIPW-corrected estimator (ate_estimator_experiment_i).
MCSE of the MSE: sd of the squared errors over the N_REPS replications / sqrt(N_REPS) (column mcse_mse).
MCSE of the bias: sd of the estimates / sqrt(N_REPS), with the sd recovered from summary.csv as
sqrt((MSE - bias^2) * N_REPS / (N_REPS - 1)), since MSE - bias^2 is the variance with divisor N_REPS.
Each value is rounded to the decimal of the second significant digit of its MCSE. Also prints the range of
MCSE / MSE of each MSE table, quoted in Sections "Plug-in estimates of the meta-learners" and "ATE estimators
and confidence intervals".
Reads visualisation_output/{metalearner,ate_estimator}_experiment_{1..4}/summary.csv. Writes the four
tables next to this script (tabulars, copied to Thesis/tables/ and input by AppendixSimulation.tex).
Run: python bias_mse_tables.py
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "visualisation_output"  # experiment outputs of this repository
N_REPS = 100


def value_with_mcse(value, mcse):
    """'value (MCSE)', both rounded to the decimal of the second significant digit of the MCSE."""
    decimals = max(0, 1 - math.floor(math.log10(mcse)))
    value_text = f"{value:.{decimals}f}".replace("-", "$-$")  # minus sign, not a hyphen
    return f"{value_text} ({mcse:.{decimals}f})"


def mse_and_mcse(row):
    return float(row["mse"]), float(row["mcse_mse"])


def bias_and_mcse(row):
    mse, bias = float(row["mse"]), float(row["mean_estimate"]) - float(row["actual_ate"])
    sd = math.sqrt((mse - bias**2) * N_REPS / (N_REPS - 1))
    return bias, sd / math.sqrt(N_REPS)


def write_table(experiment, key, methods, headers, statistic, out_name):
    """Writes the table of `statistic` (a function of a summary.csv row returning value and MCSE) for the
    methods (values of the column `key`) and returns the MCSE / value ratios."""
    lines = [
        "\\begin{tabular}{ll" + "r" * len(methods) + "}",
        "\\hline",
        "Exp. & Target & " + " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    ratios = []
    for exp_num in range(1, 5):
        with open(OUTPUTS / f"{experiment}_{exp_num}" / "summary.csv", newline="") as f:
            rows = {(r[key], r["test_distribution"]): r for r in csv.DictReader(f)}
        targets = list(dict.fromkeys(dist for _, dist in rows))  # in sweep order
        for i, dist in enumerate(targets):
            cells = []
            for method in methods:
                value, mcse = statistic(rows[method, dist])
                ratios.append(mcse / abs(value))
                cells.append(value_with_mcse(value, mcse))
            alpha = dist[5]  # "Beta(alpha,10-alpha)" -> alpha
            exp_label = f"E{exp_num}" if i == 0 else ""
            lines.append(f"{exp_label} & $\\mathrm{{Beta}}({alpha},{10 - int(alpha)})$ & "
                         + " & ".join(cells) + " \\\\")
        if exp_num < 4:
            lines[-1] += "[3pt]"  # small gap between the experiments
    lines += ["\\hline", "\\end{tabular}"]
    table = "\n".join(lines) + "\n"

    out = HERE / out_name
    out.write_text(table, encoding="utf-8")
    print(table)
    print(f"Saved to {out}\n")
    return ratios


meta = ("metalearner_experiment", "learner", ["T-learner", "X-learner", "DR-learner"],
        ["T-learner", "X-learner", "DR-learner"])
ate = ("ate_estimator_experiment", "estimator", ["Plug-in (naive CI)", "AIPW-corrected"],
       ["Plug-in", "AIPW-corrected"])

write_table(*meta, bias_and_mcse, "meta_bias.tex")
ratios = write_table(*meta, mse_and_mcse, "meta_mse.tex")
print(f"MCSE / MSE between {min(ratios):.2f} and {max(ratios):.2f}\n")

write_table(*ate, bias_and_mcse, "ate_bias.tex")
ratios = write_table(*ate, mse_and_mcse, "ate_mse.tex")
print(f"MCSE / MSE between {min(ratios):.2f} and {max(ratios):.2f}\n")
