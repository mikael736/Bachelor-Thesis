"""Table for Appendix "Additional Simulation Results", section on the variance of the AIPW-corrected
estimator: at the three central target distributions Beta(alpha, 10 - alpha), alpha = 4, 5, 6, of each
experiment E1-E4, the variance of the AIPW-corrected estimate across the 100 replications, the mean
estimated variance s_q^2 / n + s_T^2 / m, and the same estimate with the true residuals Y - mu_A(X),
with their ratios to the empirical variance. Reads visualisation_output/ate_estimator_experiment_{1..4}/diagnostics.csv.
Writes aipw_variance.tex next to this script
(tabular, input by AppendixSimulation.tex). Run: python aipw_variance_table.py
"""
import csv
from pathlib import Path

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
