"""ATE estimators for transporting the T-learner's CATE from a source sample to target covariate
samples: the plug-in estimate (average of the CATE predictions over the target), and the AIPW-corrected
estimate (Zeng, Kennedy et al. 2024, Sec. 4.2; methods note ate_CI.pdf), each with its 95% CI SE.
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from base_learners import get_classifier, get_regressor


def transport_t_learner(
    x,
    y,
    a,
    x_targets,
    *,
    n_folds: int = 5,
    propensity_clip: float = 0.05,
    max_density_ratio: float,
    regressor_params: dict | None = None,
    classifier_params: dict | None = None,
    density_ratio_params: dict | None = None,
    random_state: int,
    mu0_true=None,
    mu1_true=None,
) -> list[dict]:
    """Cross-fitted T-learner transported to each target sample in x_targets (list of (m, d) arrays).

    Per fold k (source folds stratified by a, target folds random), fitted without fold k:
    mu0, mu1, e on the source (shared across targets) and the source-vs-target classifier
    rho(x) = P(source | x) per target, with density ratio r = (n_tr / m_tr) (1 - rho) / rho, capped at
    max_density_ratio (independent of the sample sizes, unlike a floor on rho).
    classifier_params set the propensity forest, density_ratio_params the source-vs-target forest.
    On the held-out folds: t_j = tau_hat(x_j) for target units and the correction
    q_i = r(x_i) [a_i / e(x_i) (y_i - mu1(x_i)) - (1 - a_i) / (1 - e(x_i)) (y_i - mu0(x_i))].

    Returns one dict per target:
        plug_in = mean(t), se_plug_in = sd(t) / sqrt(m) (naive: target sampling only),
        aipw = mean(t) + mean(q), se_aipw = sqrt(var(q) / n + var(t) / m),
        ratio_capped_share = share of source units whose density ratio was capped,
        t_bar, q_bar, var_t_bar = var(t) / m, var_q_bar = var(q) / n (the two parts of se_aipw^2, so the
        experiment can compare them with the variances of t_bar and q_bar across replications).

    Diagnostic (simulation only): with the true outcome regressions mu0_true, mu1_true (callables on an
    (n, d) array), also se_aipw_oracle_resid - se_aipw with q computed from the true residuals
    y - mu_a(x) (e_hat and r_hat unchanged). If the AIPW interval is conservative because the
    fitting error of mu_hat inflates var(q), this SE is smaller and closer to the actual SD.
    """
    x, y, a = np.asarray(x), np.asarray(y), np.asarray(a)
    n = len(y)
    oracle = mu0_true is not None and mu1_true is not None
    if oracle:
        mu0_x, mu1_x = mu0_true(x), mu1_true(x)
    source_folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(x, a))

    # outcome and propensity models per source fold - they do not depend on the target
    reg = dict(random_state=random_state, rf_params=regressor_params)
    nuisances = []
    for train_idx, _ in source_folds:
        x_tr, y_tr, a_tr = x[train_idx], y[train_idx], a[train_idx]
        mu0 = get_regressor("rf", **reg).fit(x_tr[a_tr == 0], y_tr[a_tr == 0])
        mu1 = get_regressor("rf", **reg).fit(x_tr[a_tr == 1], y_tr[a_tr == 1])
        e = get_classifier("rf", random_state=random_state, rf_params=classifier_params).fit(x_tr, a_tr)
        nuisances.append((mu0, mu1, e))

    results = []
    for x_target in x_targets:
        m = len(x_target)
        target_folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(x_target)
        t, q, capped = np.empty(m), np.empty(n), np.empty(n, dtype=bool)
        q_oracle = np.empty(n)
        for (s_train, s_held), (t_train, t_held), (mu0, mu1, e) in zip(source_folds, target_folds, nuisances):
            # density ratio from a source (1) vs. target (0) classifier, via Bayes' rule with the training counts
            x_pool = np.vstack([x[s_train], x_target[t_train]])
            s = np.r_[np.ones(len(s_train), dtype=int), np.zeros(len(t_train), dtype=int)]
            rho_model = get_classifier("rf", random_state=random_state, rf_params=density_ratio_params).fit(x_pool, s)
            rho = rho_model.predict_proba(x[s_held])[:, 1]
            with np.errstate(divide="ignore"):  # rho = 0 gives r = inf, capped below
                r = len(s_train) / len(t_train) * (1 - rho) / rho
            capped[s_held] = r > max_density_ratio
            r = np.minimum(r, max_density_ratio)

            x_held, y_held, a_held = x[s_held], y[s_held], a[s_held]
            e_hat = np.clip(e.predict_proba(x_held)[:, 1], propensity_clip, 1 - propensity_clip)
            q[s_held] = r * (
                a_held / e_hat * (y_held - mu1.predict(x_held))
                - (1 - a_held) / (1 - e_hat) * (y_held - mu0.predict(x_held))
            )
            t[t_held] = mu1.predict(x_target[t_held]) - mu0.predict(x_target[t_held])
            if oracle:
                q_oracle[s_held] = r * (
                    a_held / e_hat * (y_held - mu1_x[s_held])
                    - (1 - a_held) / (1 - e_hat) * (y_held - mu0_x[s_held])
                )

        var_t_bar, var_q_bar = t.var(ddof=1) / m, q.var(ddof=1) / n
        result = {
            "plug_in": t.mean(),
            "se_plug_in": np.sqrt(var_t_bar),
            "aipw": t.mean() + q.mean(),
            "se_aipw": np.sqrt(var_q_bar + var_t_bar),
            "ratio_capped_share": capped.mean(),
            "t_bar": t.mean(),
            "q_bar": q.mean(),
            "var_t_bar": var_t_bar,
            "var_q_bar": var_q_bar,
        }
        if oracle:
            result["se_aipw_oracle_resid"] = np.sqrt(q_oracle.var(ddof=1) / n + var_t_bar)
        results.append(result)
    return results
