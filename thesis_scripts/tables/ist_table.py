"""Descriptive table of the IST data for Section "Data" of Chapter 6: covariates and
six-month outcome per target country, the 33 never-target countries and all countries (patients with
outcome only, as in the experiment). Uses the loader of the code repository (base_code/ist_data.py).
Writes ist_data.tex next to this script (tabular,
input by IST.tex). Run: python ist_table.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "base_code"))

import numpy as np

import ist_data

TARGET_COUNTRIES = ["UK", "ITAL", "SWIT"]
COLUMN_NAMES = ["UK", "Italy", "Switzerland", "Other", "All"]

data = ist_data.load_ist()
col = {name: data.x[:, i] for i, name in enumerate(data.feature_names)}
groups = [data.country == c for c in TARGET_COUNTRIES]
groups += [~np.isin(data.country, TARGET_COUNTRIES), np.ones(len(data.y), dtype=bool)]


def percent(indicator, among=None, decimals=0):
    """Share in % per group, among the patients with among == 1 if given (e.g. value recorded)."""
    among = np.ones(len(indicator), dtype=bool) if among is None else among.astype(bool)
    shares = [100 * indicator[g & among].mean() for g in groups]
    return ["$<1$" if 0 < share < 0.5 else f"{share:.{decimals}f}" for share in shares]


def mean_sd(values):
    return [f"{values[g].mean():.0f} ({values[g].std(ddof=1):.0f})" for g in groups]


def median_iqr(values):
    return [f"{np.median(values[g]):.0f} ({np.percentile(values[g], 25):.0f}--{np.percentile(values[g], 75):.0f})"
            for g in groups]


def risk_difference(g):
    """Risk difference (aspirin - no aspirin) in percentage points and its Wald 95% CI."""
    y1, y0 = data.y[g & (data.a == 1)], data.y[g & (data.a == 0)]
    p1, p0 = y1.mean(), y0.mean()
    rd, se = p1 - p0, np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))
    rd_text = f"{100 * rd:.1f}".replace("-0.0", "0.0")  # UK: -0.04 rounds to 0.0, not -0.0
    return f"${rd_text}$", f"$\\pm{196 * se:.1f}$"


risk_differences = [risk_difference(g) for g in groups]
rows = [
    # units and summaries (mean (SD), median (IQR), shares in %) are given in the caption, so that the
    # only parentheses in a row are the SD or IQR
    ("Patients", [f"{g.sum()}" for g in groups]),
    ("Allocated to aspirin", percent(data.a)),
    (None, "Covariates"),
    ("Age", mean_sd(col["AGE"])),
    ("Female", percent(col["SEX_F"])),
    ("Systolic blood pressure", mean_sd(col["RSBP"])),
    ("Delay to randomization", median_iqr(col["RDELAY"])),
    ("Drowsy or unconscious", percent(col["RCONSC_D"] + col["RCONSC_U"])),
    ("Atrial fibrillation$^a$", percent(col["RATRIAL_Y"], 1 - col["RATRIAL_missing"])),
    ("Aspirin in the previous 3 days$^a$", percent(col["RASP3_Y"], 1 - col["RASP3_missing"])),
    ("Stroke subtype: total anterior", percent(col["STYPE_TACS"])),
    ("\\quad partial anterior", percent(col["STYPE_PACS"])),
    ("\\quad lacunar", percent(col["STYPE_LACS"])),
    ("\\quad posterior", percent(col["STYPE_POCS"])),
    ("\\quad other", percent(col["STYPE_OTH"])),
    (None, "Dead or dependent at six months"),
    # one decimal, so that the risk difference below follows from these two rows
    ("With aspirin", percent(data.y, data.a == 1, decimals=1)),
    ("Without aspirin", percent(data.y, data.a == 0, decimals=1)),
    ("Risk difference (pp)", [rd for rd, _ in risk_differences]),
    ("\\quad 95\\% CI half-width", [hw for _, hw in risk_differences]),
]

# plain \hline rules as in the other tables of the thesis (the template loads no table packages)
n_cols = len(COLUMN_NAMES)
lines = [
    f"\\begin{{tabular}}{{l{'r' * n_cols}}}",
    "\\hline",
    " & " + " & ".join(COLUMN_NAMES) + " \\\\",
    "\\hline",
]
for label, cells in rows:
    if label is None:
        # [3pt]: small gap above each block heading
        lines[-1] += "[3pt]"
        lines.append(f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{{cells}}}}} \\\\")
    else:
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
lines += ["\\hline", "\\end{tabular}"]
table = "\n".join(lines) + "\n"

out = Path(__file__).resolve().parent / "ist_data.tex"
out.write_text(table, encoding="utf-8")
print(table)
print(f"Saved to {out}")
