"""IST real-data experiment: transport the aspirin effect to a held-out country. For each target
country, fit the T-, X- and DR-learner on all other countries, average the predicted CATE over the
target's covariates (plug-in estimate), and compare with the target's own randomized estimate.
Second part, as in the ATE-estimator experiment: the cross-fitted T-learner's plug-in estimate with
its naive CI and the AIPW-corrected estimate with its CI.
Design and data preparation: experiment_design.md ("IST real-data experiment").
"""
import csv
import sys
from pathlib import Path

# snapshot this script at launch, before the long run - the file may be edited for other runs meanwhile
SOURCE_SNAPSHOT = Path(__file__).read_text(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "base_code"))

import numpy as np

import ate_estimators
import cate_learners
import ist_data
import save_run

# -----------------------------------------------------------------------------
# 0. Run settings - part of the script, so the saved snapshot records them
# -----------------------------------------------------------------------------

SEED = 42
TARGET_COUNTRIES = ["UK", "ITAL", "SWIT"]  # the three largest by enrolment
# tuned by OOB Brier score on the never-target countries (ist_tune_rf.py)
REGRESSOR_PARAMS = dict(max_features=1 / 3, min_samples_leaf=20)
CLASSIFIER_PARAMS = dict(max_features="sqrt", min_samples_leaf=100)
# AIPW-corrected estimator: settings as in ate_estimator_experiment.py; the source-vs-target forest
# for the density ratio as there (leaf 100, not tuned on IST)
N_FOLDS = 5
MAX_DENSITY_RATIO = 20
PROPENSITY_CLIP = 0.05
DENSITY_RATIO_PARAMS = dict(max_features="sqrt", min_samples_leaf=100)

rf = dict(base_learner="rf", regressor_params=REGRESSOR_PARAMS, random_state=SEED)
learners = [
    ("T-learner", lambda: cate_learners.TLearner(**rf)),
    ("X-learner", lambda: cate_learners.XLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
    ("DR-learner", lambda: cate_learners.DRLearner(classifier_params=CLASSIFIER_PARAMS, **rf)),
]
run_tag = "ist_experiment"

data = ist_data.load_ist()  # patients with a missing outcome are excluded


def risk_difference(y, a):
    """Difference in outcome means (aspirin - no aspirin), its SE and the arm sizes."""
    y1, y0 = y[a == 1], y[a == 0]
    p1, p0 = y1.mean(), y0.mean()
    return p1 - p0, np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0)), len(y1), len(y0)


# -----------------------------------------------------------------------------
# 1. Transport: fit on the source (other countries), average the CATE over the target
# patients' covariates. Target outcomes are not used here.
# -----------------------------------------------------------------------------

estimates, ate_results = {}, {}
for country in TARGET_COUNTRIES:
    source = data.country != country
    x_target = data.x[data.country == country]
    estimates[country] = {
        tag: make_learner().fit(data.x[source], data.y[source], data.a[source]).predict(x_target).mean()
        for tag, make_learner in learners
    }
    print(f"{country}: " + ", ".join(f"{tag} {100 * est:.2f} pp" for tag, est in estimates[country].items()))
    # cross-fitted T-learner: plug-in (naive CI) and AIPW-corrected estimate (CI), as in Chapter 4
    ate_results[country] = ate_estimators.transport_t_learner(
        data.x[source], data.y[source], data.a[source], [x_target],
        n_folds=N_FOLDS, propensity_clip=PROPENSITY_CLIP, max_density_ratio=MAX_DENSITY_RATIO,
        regressor_params=REGRESSOR_PARAMS, classifier_params=CLASSIFIER_PARAMS,
        density_ratio_params=DENSITY_RATIO_PARAMS, random_state=SEED,
    )[0]
    result = ate_results[country]
    print(f"{country}: plug-in {100 * result['plug_in']:.2f} +- {196 * result['se_plug_in']:.2f} pp, "
          f"AIPW {100 * result['aipw']:.2f} +- {196 * result['se_aipw']:.2f} pp, "
          f"capped ratios {100 * result['ratio_capped_share']:.1f}%")

# -----------------------------------------------------------------------------
# 2. Reveal: each target's randomized estimate with its 95% CI (values in percentage points)
# -----------------------------------------------------------------------------

rows = []
for country in TARGET_COUNTRIES:
    target = data.country == country
    rd, se, n1, n0 = risk_difference(data.y[target], data.a[target])
    result = ate_results[country]
    ate_columns = {}
    for key in ("plug_in", "aipw"):
        ate_columns[key] = 100 * result[key]
        ate_columns[f"{key}_ci_low"] = 100 * (result[key] - 1.96 * result[f"se_{key}"])
        ate_columns[f"{key}_ci_high"] = 100 * (result[key] + 1.96 * result[f"se_{key}"])
    rows.append({
        "country": country,
        "n_source": int((~target).sum()),
        "n_target": int(target.sum()),
        "n_target_aspirin": n1,
        "n_target_control": n0,
        "source_rd_untransported": 100 * risk_difference(data.y[~target], data.a[~target])[0],
        **{tag: 100 * est for tag, est in estimates[country].items()},
        **ate_columns,
        "ratio_capped_share": result["ratio_capped_share"],
        "benchmark_rd": 100 * rd,
        "benchmark_ci_low": 100 * (rd - 1.96 * se),
        "benchmark_ci_high": 100 * (rd + 1.96 * se),
    })

# -----------------------------------------------------------------------------
# 3. Save snapshot, table and plot to visualisation_output/<run_tag>/
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visualisation_code"))
from plotting import plot_ist, plot_ist_ci

output_root = Path(__file__).resolve().parent.parent / "visualisation_output"
run_dir = save_run.save_run(run_tag, SOURCE_SNAPSHOT, output_root=output_root)
with open(run_dir / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plot_ist(rows, [tag for tag, _ in learners], run_dir / "plot.png")
plot_ist_ci(rows, run_dir / "ci_plot.png")
print(f"Saved to {run_dir}")
