"""Tune random-forest max_features and min_samples_leaf for the IST experiment by out-of-bag (OOB)
Brier score of the nuisance models mu0, mu1 (outcome per arm) and e (aspirin allocation).
Uses only the countries that are never a target, so the chosen setting is source-only for all
three targets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import ist_data

SEED = 42
TARGET_COUNTRIES = ["UK", "ITAL", "SWIT"]
MAX_FEATURES = ["sqrt", 1 / 3, 0.5, 1.0]
LEAF_SIZES = [5, 20, 50, 100]

data = ist_data.load_ist()
keep = ~np.isin(data.country, TARGET_COUNTRIES)
x, a, y = data.x[keep], data.a[keep], data.y[keep]
print(f"tuning data: {keep.sum()} patients from {len(set(data.country[keep]))} countries")

rows = []
for max_features in MAX_FEATURES:
    for leaf in LEAF_SIZES:
        kwargs = dict(max_features=max_features, min_samples_leaf=leaf, oob_score=True, random_state=SEED)
        briers = []
        for arm in (0, 1):
            fit = a == arm
            rf = RandomForestRegressor(**kwargs).fit(x[fit], y[fit])
            briers.append(np.mean((rf.oob_prediction_ - y[fit]) ** 2))
        clf = RandomForestClassifier(**kwargs).fit(x, a)
        briers.append(np.mean((clf.oob_decision_function_[:, 1] - a) ** 2))
        rows.append((max_features, leaf, *briers))

# OOB Brier per model; best value per column marked with *
best = np.array([r[2:] for r in rows]).argmin(axis=0)
print(f"{'max_features':>12}{'leaf':>6}{'mu0':>10}{'mu1':>10}{'e':>10}")
for i, (max_features, leaf, *briers) in enumerate(rows):
    label = max_features if isinstance(max_features, str) else f"{max_features:.2f}"
    cells = "".join(f"{b:>9.4f}{'*' if i == best[j] else ' '}" for j, b in enumerate(briers))
    print(f"{label:>12}{leaf:>6}{cells}")
