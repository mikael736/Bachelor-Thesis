"""Table of the IST estimates behind Figures fig:ist and fig:ist-ci (Section "Results" of Chapter 6):
per target country the randomized estimate, the untransported study estimate, the transported
plug-in estimates of the three meta-learners, and the plug-in (naive CI) and AIPW-corrected (CI)
estimates of the cross-fitted T-learner, in percentage points, each interval as its 95% CI. Reads
visualisation_output/ist_experiment/results.csv. Writes ist_results.tex next to this script (tabular,
input by IST.tex).
Run: python ist_results_table.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "visualisation_output"  # experiment outputs of this repository
COUNTRY_NAMES = {"UK": "UK", "ITAL": "Italy", "SWIT": "Switzerland"}

with open(OUTPUTS / "ist_experiment" / "results.csv", newline="") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    half_width = (float(r["source_rd_ci_high"]) - float(r["source_rd_ci_low"])) / 2
    print(f"{r['country']}: half-width of the study interval {half_width:.2f} pp")


def number(value):
    """Two decimals in math mode; -0.00 printed as 0.00."""
    return f"${float(value):.2f}$".replace("$-0.00$", "$0.00$")


def interval(row, low, high):
    """The 95% interval on its own row below the estimate, so that the table fits the text width."""
    return f"({number(row[low])}, {number(row[high])})"


lines_spec = [
    ("Benchmark", lambda r: number(r["benchmark_rd"])),
    ("\\quad 95\\% normal-based interval", lambda r: interval(r, "benchmark_ci_low", "benchmark_ci_high")),
    ("Study, not transported", lambda r: number(r["source_rd_untransported"])),
    ("\\quad 95\\% normal-based interval", lambda r: interval(r, "source_rd_ci_low", "source_rd_ci_high")),
    (None, "Transported plug-in estimates (Figure~\\ref{fig:ist})"),
    ("T-learner", lambda r: number(r["T-learner"])),
    ("X-learner", lambda r: number(r["X-learner"])),
    ("DR-learner", lambda r: number(r["DR-learner"])),
    (None, "ATE estimators with confidence intervals (Figure~\\ref{fig:ist-ci})"),
    ("Plug-in", lambda r: number(r["plug_in"])),
    ("\\quad naive interval", lambda r: interval(r, "plug_in_ci_low", "plug_in_ci_high")),
    ("AIPW-corrected", lambda r: number(r["aipw"])),
    ("\\quad AIPW interval", lambda r: interval(r, "aipw_ci_low", "aipw_ci_high")),
]

n_cols = len(rows)
lines = [
    f"\\begin{{tabular}}{{l{'r' * n_cols}}}",
    "\\hline",
    " & " + " & ".join(COUNTRY_NAMES[r["country"]] for r in rows) + " \\\\",
    "\\hline",
]
for label, cells in lines_spec:
    if label is None:
        lines[-1] += "[3pt]"  # small gap above each block heading
        lines.append(f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{{cells}}}}} \\\\")
    else:
        lines.append(f"{label} & " + " & ".join(cells(r) for r in rows) + " \\\\")
lines += ["\\hline", "\\end{tabular}"]
table = "\n".join(lines) + "\n"

out = HERE / "ist_results.tex"
out.write_text(table, encoding="utf-8")
print(table)
print(f"Saved to {out}")
