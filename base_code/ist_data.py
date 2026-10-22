"""Load and prepare the International Stroke Trial (IST) data (Sandercock et al. 2011, version 2,
IST_corrected.csv) for the real-data experiment: 16 covariates recorded before randomization, the
categorical ones one-hot encoded with a level of their own for missing values; patients without a
recorded six-month outcome are excluded.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "ist" / "IST_corrected.csv"

NUMERIC = ["AGE", "RSBP", "RDELAY"]
# categorical covariates with their codebook levels; "" is the missing level (RATRIAL/RASP3 in the pilot)
CATEGORICAL = {
    "SEX": ["M", "F"],
    "RCONSC": ["F", "D", "U"],
    "RATRIAL": ["Y", "N", ""],
    "RASP3": ["Y", "N", ""],
    **{f"RDEF{i}": ["Y", "N", "C"] for i in range(1, 9)},
    "STYPE": ["TACS", "PACS", "POCS", "LACS", "OTH"],
}
# OCCODE: 1 dead, 2 dependent -> 1 (death or dependency); 3 not recovered, 4 recovered -> 0; else missing (excluded)
OUTCOME_CODES = {"1": 1.0, "2": 1.0, "3": 0.0, "4": 0.0}


@dataclass(frozen=True)
class IstData:
    x: np.ndarray  # (n, p) covariates: numeric columns, then one-hot categorical columns
    feature_names: list[str]
    a: np.ndarray  # (n,) 1 = aspirin allocated
    y: np.ndarray  # (n,) 1 = dead or dependent at six months
    country: np.ndarray  # (n,) country code, for splitting only (not a covariate)


def load_ist(path: Path = DEFAULT_PATH) -> IstData:
    """Patients with a missing outcome (150, 0.8%) are excluded."""
    with open(path, newline="", encoding="latin-1") as f:
        rows = [r for r in csv.DictReader(f) if r["OCCODE"].strip() in OUTCOME_CODES]

    columns, feature_names = [], []
    for name in NUMERIC:
        columns.append([float(r[name]) for r in rows])
        feature_names.append(name)
    for name, levels in CATEGORICAL.items():
        values = [r[name].strip() for r in rows]
        unexpected = set(values) - set(levels)
        if unexpected:
            raise ValueError(f"{name}: values {unexpected} not in codebook levels {levels}")
        for level in levels:
            columns.append([float(v == level) for v in values])
            feature_names.append(f"{name}_{level or 'missing'}")

    return IstData(
        x=np.array(columns).T,
        feature_names=feature_names,
        a=np.array([{"Y": 1, "N": 0}[r["RXASP"]] for r in rows]),
        y=np.array([OUTCOME_CODES[r["OCCODE"].strip()] for r in rows]),
        country=np.array([r["COUNTRY"] for r in rows]),
    )
