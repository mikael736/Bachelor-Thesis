# Scripts for the thesis text

These scripts produce the tables and the numbers quoted in the thesis from the experiment outputs in
`visualisation_output/` (written by the scripts in `running_code/`). Run each script from its own folder,
e.g. `cd thesis_scripts/numbers && python ate_estimator_comparison.py`.

## `tables/`: LaTeX tables

Each script writes its `.tex` table next to itself; the tables are copied to the thesis.

| Script | Table | Thesis |
|---|---|---|
| `bias_mse_tables.py` | `meta_bias.tex`, `meta_mse.tex`, `ate_bias.tex`, `ate_mse.tex` | Appendix C.1 |
| `aipw_variance_table.py` | `aipw_variance.tex` | Appendix C.4 |
| `ist_table.py` | `ist_data.tex` | Section 6.1 |
| `ist_results_table.py` | `ist_results.tex` | Section 6.3 |

## `numbers/`: numbers quoted in the text

Each script prints the numbers of one part of the thesis; its docstring lists them.

| Script | Thesis |
|---|---|
| `meta_learner_comparison.py` | Section 5.2, meta-learner paragraph of Chapter 7 |
| `ate_estimator_comparison.py` | Section 5.3, Appendix C.2, ATE-estimator paragraph of Chapter 7 |
| `density_ratio_cap.py` | introduction of Section 5.3 (how often the cap on the density ratio is reached) |
| `tail_masses.py` | Section 5.1 (tail masses, propensity range), Section 5.2 (units per arm in the tails) |
| `response_variances.py` | Table 5.1 (variances of the response surfaces) |
| `density_ratio_moments.py` | Lemma C.1 and Table C.5 (second moment of the density ratio) |

`summaries.py` is a shared helper that reads the `summary.csv` files; it prints nothing.
