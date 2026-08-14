from abc import ABC, abstractmethod

import numpy as np

import config
from base_learners import get_classifier, get_regressor


class BaseCATELearner(ABC):
    """Common interface for all CATE learners, meta-learner or not."""

    @abstractmethod
    def fit(self, x, y, a):
        ...

    @abstractmethod
    def predict(self, x):
        ...


class TLearner(BaseCATELearner):
    """T-learner: the simplest meta-learner, two independent response-surface fits.

    mu0 fit on control units, mu1 fit on treated units, no propensity model at all:
        tau_hat(x) = mu1_hat(x) - mu0_hat(x)
    """

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


class XLearner(BaseCATELearner):
    """X-learner (Kuenzel et al., 2019): a pseudo-outcome method that reuses each arm's fitted
    model to impute the other arm's missing potential outcome, then regresses the resulting
    per-unit treatment-effect estimates on x, separately per arm.
    """

    def __init__(
        self,
        base_learner: str = "rf",
        mu0_learner: str | None = None,
        propensity_learner: str = "rf",
        *,
        random_state: int = config.SEED,
    ):
        """mu0_learner, if given, overrides base_learner just for mu0 - e.g. a correctly-specified
        parametric model for the control/natural-history surface, while mu1, tau0, and tau1 stay
        on the flexible base_learner. Defaults to base_learner (no override).
        """
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

        # Step 1 (same as T-learner): fit mu0 on control units, mu1 on treated units
        self.mu0_model.fit(x[a == 0], y[a == 0])
        self.mu1_model.fit(x[a == 1], y[a == 1])

        # Step 2: impute the missing potential outcome with the other arm's model, forming
        # per-unit pseudo-treatment-effects (Equation 8)
        d1 = y[a == 1] - self.mu0_model.predict(x[a == 1])  # treated units: observed Y1 - imputed mu0
        d0 = self.mu1_model.predict(x[a == 0]) - y[a == 0]  # control units: imputed mu1 - observed Y0

        # Step 3: regress each arm's pseudo-outcome on x, giving tau0_hat and tau1_hat
        self.tau0_model.fit(x[a == 0], d0)
        self.tau1_model.fit(x[a == 1], d1)

        # Step 4: estimate the propensity score, used to weight the two CATE estimates
        self.propensity_model.fit(x, a)
        return self

    def predict(self, x):
        x = np.asarray(x)
        e_hat = self.propensity_model.predict_proba(x)[:, 1]
        tau0_hat = self.tau0_model.predict(x)
        tau1_hat = self.tau1_model.predict(x)
        return e_hat * tau0_hat + (1 - e_hat) * tau1_hat
