import numpy as np

import config
from base_learners import get_classifier, get_regressor


class TLearner:
    """T-learner."""

    def __init__(self, base_learner: str = "rf", *, random_state: int = config.SEED):
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
        random_state: int = config.SEED,
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
    """Doubly robust learner (Kennedy, 2020)."""

    def __init__(
        self,
        base_learner: str = "rf",
        propensity_learner: str = "rf",
        *,
        random_state: int = config.SEED,
    ):
        self.base_learner = base_learner
        self.propensity_learner = propensity_learner
        self.name = f"DRLearner[{base_learner}]"
        self.mu0_model = get_regressor(base_learner, random_state=random_state)
        self.mu1_model = get_regressor(base_learner, random_state=random_state)
        self.propensity_model = get_classifier(propensity_learner, random_state=random_state)
        self.tau_model = get_regressor(base_learner, random_state=random_state)

    def fit(self, x, y, a):
        x, y, a = np.asarray(x), np.asarray(y), np.asarray(a)

        # Step 1: fit mu0/mu1 and e (as in T-learner + X-learner's propensity step)
        self.mu0_model.fit(x[a == 0], y[a == 0])
        self.mu1_model.fit(x[a == 1], y[a == 1])
        self.propensity_model.fit(x, a)

        # Step 2: AIPW pseudo-outcome, doubly robust to mu-model or propensity-model error.
        # Clip e_hat away from 0/1 - a classifier can be fully confident on some units, which
        # would otherwise divide by zero below.
        e_hat = np.clip(self.propensity_model.predict_proba(x)[:, 1], 1e-3, 1 - 1e-3)
        mu0_hat = self.mu0_model.predict(x)
        mu1_hat = self.mu1_model.predict(x)
        phi = (mu1_hat - mu0_hat) + a / e_hat * (y - mu1_hat) - (1 - a) / (1 - e_hat) * (y - mu0_hat)

        # Step 3: regress the pseudo-outcome on x -> tau_hat
        self.tau_model.fit(x, phi)
        return self

    def predict(self, x):
        x = np.asarray(x)
        return self.tau_model.predict(x)
