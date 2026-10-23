"""Table for Appendix "Additional Simulation Results", section on the variance of the AIPW-corrected
estimator: at the three central target distributions Beta(alpha, 10 - alpha), alpha = 4, 5, 6, of each
experiment E1-E4, the variance of the AIPW-corrected estimate across the 100 replications, the mean
estimated variance s_q^2 / n + s_T^2 / m, and the same estimate with the true residuals Y - mu_A(X),
with their ratios to the empirical variance. Reads
visualisation_output/ate_estimator_experiment_{1..4}/diagnostics.csv. Writes aipw_variance.tex next to
this script (tabular, input by AppendixSimulation.tex). Run: python aipw_variance_table.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy import integrate, stats

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "visualisation_output"  # experiment outputs of this repository
CENTRAL = ["Beta(4,6)", "Beta(5,5)", "Beta(6,4)"]

lines = [
    "\\begin{tabular}{llrrrrr}",
    "\\hline",
    " & & & \\multicolumn{2}{c}{Estimated} & \\multicolumn{2}{c}{True residuals} \\\\",
    "Exp. & Target & Empirical & Variance & Ratio & Variance & Ratio \\\\",
    "\\hline",
]
for exp_num in range(1, 5):
    with open(OUTPUTS / f"ate_estimator_experiment_{exp_num}" / "diagnostics.csv", newline="") as f:
        rows = {r["test_distribution"]: r for r in csv.DictReader(f)}
    for i, dist in enumerate(CENTRAL):
        r = rows[dist]
        empirical, est, oracle = float(r["var_aipw"]), float(r["est_var_aipw"]), float(r["est_var_aipw_oracle_resid"])
        k = dist[5]  # "Beta(alpha,10-alpha)" -> alpha
        exp_label = f"E{exp_num}" if i == 0 else ""
        lines.append(f"{exp_label} & $\\mathrm{{Beta}}({k},{10 - int(k)})$ & {empirical:.3f} & {est:.3f} & "
                     f"{est / empirical:.2f} & {oracle:.3f} & {oracle / empirical:.2f} \\\\")
    if exp_num < 4:
        lines[-1] += "[3pt]"  # small gap between the experiments
lines += ["\\hline", "\\end{tabular}"]
table = "\n".join(lines) + "\n"

out = HERE / "aipw_variance.tex"
out.write_text(table, encoding="utf-8")
print(table)
print(f"Saved to {out}")

# numbers quoted in Appendix C.4 and in the DR-learner paragraph of Section 5.2, at Beta(5,5) (r = 1):
# E[1/(e(X)(1 - e(X)))] under the study distribution Beta(5,5) (variance of q* by Remark C.2, and the
# conditional variance of the DR pseudo-outcome with true nuisances), divided by n = 1000 for the mean
# of the correction terms; and the part of the true-residual estimate that belongs to q_bar
e = lambda x: 1 / (1 + np.exp(-5 * (x - 0.5)))
moment = integrate.quad(lambda x: stats.beta(5, 5).pdf(x) / (e(x) * (1 - e(x))), 0, 1)[0]
print(f"\nE[1/(e(1-e))] under Beta(5,5): {moment:.2f}; divided by n = 1000: {moment / 1000:.4f}")
for exp_num in range(1, 5):
    with open(OUTPUTS / f"ate_estimator_experiment_{exp_num}" / "diagnostics.csv", newline="") as f:
        r = {r["test_distribution"]: r for r in csv.DictReader(f)}["Beta(5,5)"]
    print(f"E{exp_num}, Beta(5,5): q_bar part of the true-residual estimate "
          f"{float(r['est_var_aipw_oracle_resid']) - float(r['est_var_t_bar']):.4f}")
