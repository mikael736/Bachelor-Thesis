"""How often the estimated density ratio reaches the cap R, per target distribution Beta(a, 10 - a), quoted
in the introduction of Section "ATE estimators and confidence intervals" (Chapter 5). Reads the per-replication
share of capped study units (ratio_capped_share) from
visualisation_output/ate_estimator_experiment_{1..4}/estimates.npz. The share depends only on the covariates,
so it is the same in all four experiments; the script checks this.
Run: python density_ratio_cap.py
"""
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[2] / "visualisation_output"
N_STUDY = 1000
TARGETS = [f"Beta({a},{10 - a})" for a in range(2, 9)]

shares = [np.load(OUTPUTS / f"ate_estimator_experiment_{i}" / "estimates.npz")["ratio_capped_share"]
          for i in range(1, 5)]  # each (N_REPS, n_targets)
print(f"same in all four experiments: {all(np.array_equal(s, shares[0]) for s in shares)}")
share = shares[0]
print(f"{'target':12s}{'capped units per replication (mean)':>38s}{'replications with a capped unit':>34s}")
for j, target in enumerate(TARGETS):
    print(f"{target:12s}{N_STUDY * share[:, j].mean():38.2f}{(share[:, j] > 0).mean():34.0%}")
