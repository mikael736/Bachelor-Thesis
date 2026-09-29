import numpy as np
from sklearn.model_selection import StratifiedKFold

from base_learners import get_classifier, get_regressor


class TLearner:
    """T-learner."""

    def __init__(self, base_learner: str = "rf", *, random_state: int):
        self.base_learner = base_learner
        self.name = f"TLearner[{base_learner}]"
        self.mu0_model = get_regressor(base_learner, random_state=random_state)
        self.mu1_model = get_regressor(base_learner, random_state=random_state)

    def fit(self, x, y, a):
        x, y, a = np.asarray(x), np.asarray(y), np.asarray(a)
        self.mu0_model.fit(x[a == 0], y[a == 0])
        self.mu1_model.fit(x[a == 1], y[a == 1])
        return self

    def predict(self, x):
        x = np.asarray(x)
        return self.mu1_model.predict(x) - self.mu0_model.predict(x)


class XLearner:
    """X-learner (Kuenzel et al., 2019)."""

    def __init__(
        self,
        base_learner: str = "rf",
        mu0_learner: str | None = None,
        propensity_learner: str = "rf",
        *,
        random_state: int,
    ):
        """mu0_learner overrides base_learner just for mu0; defaults to base_learner."""
        mu0_learner = mu0_learner or base_learner
        self.base_learner = base_learner
        self.mu0_learner = mu0_learner
        self.propensity_learner = propensity_learner
        self.name = f"XLearner[{base_learner}, mu0={mu0_learner}]"
        self.mu0_model = get_regressor(mu0_learner, random_state=random_state)
        self.mu1_model = get_regressor(base_learner, random_state=random_state)
        self.tau0_model = get_regressor(base_learner, random_state=random_state)
        self.tau1_model = get_regressor(base_learner, random_state=random_state)
        self.propensity_model = get_classifier(propensity_learner, random_state=random_state)

    def fit(self, x, y, a):
        x, y, a = np.asarray(x), np.asarray(y), np.asarray(a)

        # Step 1: fit mu0/mu1 (as in T-learner)
        self.mu0_model.fit(x[a == 0], y[a == 0])
        self.mu1_model.fit(x[a == 1], y[a == 1])

        # Step 2: impute the missing potential outcome, forming per-unit pseudo-effects
        d1 = y[a == 1] - self.mu0_model.predict(x[a == 1])
        d0 = self.mu1_model.predict(x[a == 0]) - y[a == 0]

        # Step 3: regress each arm's pseudo-outcome on x -> tau0_hat, tau1_hat
        self.tau0_model.fit(x[a == 0], d0)
        self.tau1_model.fit(x[a == 1], d1)

        # Step 4: propensity score, used to weight the two CATE estimates
        self.propensity_model.fit(x, a)
        return self

    def predict(self, x):
        x = np.asarray(x)
        e_hat = self.propensity_model.predict_proba(x)[:, 1]
        tau0_hat = self.tau0_model.predict(x)
        tau1_hat = self.tau1_model.predict(x)
        return e_hat * tau0_hat + (1 - e_hat) * tau1_hat


class DRLearner:
    """Doubly robust learner (Kennedy, 2020), with K-fold cross-fitting of the nuisance models.

    cross_fit=False fits the nuisances on all units and evaluates them in-sample instead - with
    flexible base learners (e.g. rf) the residuals then nearly vanish and the pseudo-outcome
    collapses onto the T-learner's estimate; kept only as an ablation.
    """

    def __init__(
        self,
        base_learner: str = "rf",
        propensity_learner: str = "rf",
        *,
        cross_fit: bool = True,
        n_folds: int = 5,
        propensity_clip: float = 0.05,
        random_state: int,
    ):
        self.base_learner = base_learner
        self.propensity_learner = propensity_learner
        self.cross_fit = cross_fit
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state
        self.name = f"DRLearner[{base_learner}]" if cross_fit else f"DRLearner[{base_learner}, no cross-fit]"
        self.tau_model = get_regressor(base_learner, random_state=random_state)

    def _fit_nuisances(self, x, y, a):
        """Fresh mu0, mu1 (as in T-learner) and propensity e (as in X-learner) fitted on (x, y, a)."""
        mu0 = get_regressor(self.base_learner, random_state=self.random_state).fit(x[a == 0], y[a == 0])
        mu1 = get_regressor(self.base_learner, random_state=self.random_state).fit(x[a == 1], y[a == 1])
        e = get_classifier(self.propensity_learner, random_state=self.random_state).fit(x, a)
        return mu0, mu1, e

    def _pseudo_outcome(self, nuisances, x, y, a):
        """AIPW pseudo-outcome, doubly robust to mu-model or propensity-model error. e_hat is
        clipped away from 0/1 so a confident classifier can't blow up the inverse weights.
        """
        mu0, mu1, e = nuisances
        e_hat = np.clip(e.predict_proba(x)[:, 1], self.propensity_clip, 1 - self.propensity_clip)
        mu0_hat = mu0.predict(x)
        mu1_hat = mu1.predict(x)
        return (mu1_hat - mu0_hat) + a / e_hat * (y - mu1_hat) - (1 - a) / (1 - e_hat) * (y - mu0_hat)

    def fit(self, x, y, a):
        x, y, a = np.asarray(x), np.asarray(y), np.asarray(a)

        # Steps 1+2: nuisances and pseudo-outcome. Cross-fitted: each unit's phi comes from
        # nuisances fitted on the other folds, never on the unit itself. Folds are stratified by
        # a so every fold's training part keeps both arms in the original proportion.
        if self.cross_fit:
            phi = np.empty(len(y))
            folds = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            for train_idx, held_idx in folds.split(x, a):
                nuisances = self._fit_nuisances(x[train_idx], y[train_idx], a[train_idx])
                phi[held_idx] = self._pseudo_outcome(nuisances, x[held_idx], y[held_idx], a[held_idx])
        else:
            phi = self._pseudo_outcome(self._fit_nuisances(x, y, a), x, y, a)

        # Step 3: regress the pseudo-outcome on x (all units) -> tau_hat
        self.tau_model.fit(x, phi)
        return self

    def predict(self, x):
        x = np.asarray(x)
        return self.tau_model.predict(x)
