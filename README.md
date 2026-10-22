# Estimation of ATEs via CATEs

Code for the Bachelor thesis *Estimation of ATEs via CATEs* (Seminar for Statistics, ETH Zurich,
2026). The thesis estimates the
average treatment effect (ATE) in a target population from a randomized study in a different
population: CATE meta-learners (T-, X- and DR-learner) are fitted on the study sample and their
predictions averaged over the target covariates (plug-in estimator), optionally with an AIPW
correction. The estimators are compared in simulation experiments under covariate shift and applied
to the International Stroke Trial (IST).

## Structure

| Folder | Content |
|---|---|
| `base_code/` | Modules: covariate distributions, response surfaces, propensity scores, noise, base learners, meta-learners, ATE estimators, IST data loader |
| `running_code/` | Experiment and tuning scripts (see below) |
| `visualisation_code/` | `plotting.py`, the figures of the thesis |
| `visualisation_output/` | Outputs of the final experiment runs, one folder per run |
| `thesis_scripts/` | Tables and numbers quoted in the thesis, computed from `visualisation_output/` (see `thesis_scripts/README.md`) |
| `data/ist/` | IST database, version 2 (see `data/ist/README.md` for source, citation and license) |

## Setup

Python 3.11 with the pinned packages:

```
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

| Script | Thesis | Output folder |
|---|---|---|
| `metalearner_experiment.py` | Section 5.2 | `metalearner_experiment_{1..4}` |
| `ate_estimator_experiment.py` | Section 5.3, Appendix C | `ate_estimator_experiment_{1..4}` |
| `ist_experiment.py` | Chapter 6 | `ist_experiment` |

Run a script from the repository root, e.g. `python running_code/metalearner_experiment.py`. The
simulation scripts run one experiment E1–E4 per launch; the experiment is selected in part 1 of the
script (`mu0_shape`, `treatment_effect_shape`, `treatment_effect_breakpoints`, `run_tag`), where the
settings of all four experiments are listed. The scripts are set to E1.

Each run writes its figures (PDF and PNG), its tables (`summary.csv`, `diagnostics.csv` or
`results.csv`), its per-replication estimates (`estimates.npz`) and a copy of the script
(`running_code.py`) to `visualisation_output/<run_tag>/`. All random numbers derive from a fixed seed,
so a rerun reproduces the outputs exactly. A rerun with an unchanged script overwrites its folder; a
changed script with the same `run_tag` writes to `<run_tag>_2`, `<run_tag>_3`, … instead.

The tuning scripts print the out-of-bag comparisons behind the random-forest settings and write no
files:

| Script | Setting |
|---|---|
| `metalearner_tuning.py` | leaf size of the response-surface and propensity forests (Section 5.1) |
| `ate_estimator_tuning.py` | leaf size of the study-vs-target forest for the density ratio (Section 5.1) |
| `ist_tuning.py` | `max_features` and leaf size of the forests in the IST application (Chapter 6) |

## Tables and numbers of the thesis

The scripts in `thesis_scripts/` read the outputs in `visualisation_output/` and write the LaTeX tables
or print the numbers quoted in the text. Run each from its own folder, e.g.
`cd thesis_scripts/numbers && python ate_estimator_comparison.py`; `thesis_scripts/README.md` maps
each script to the part of the thesis it supports.

## Data

The IST data are distributed under the Open Data Commons Attribution License (ODC-By):
Sandercock PAG, Niewada M, Czlonkowska A, and the International Stroke Trial Collaborative Group
(2011). The International Stroke Trial database. *Trials* 12:101.
