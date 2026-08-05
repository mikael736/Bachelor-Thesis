from abc import ABC, abstractmethod

import numpy as np

from base_learners import get_regressor


class BaseCATELearner(ABC):
    """Common interface for all CATE learners, meta-learner or not."""

    @abstractmethod
    def fit(self, x, y, w):
        ...

    @abstractmethod
    def predict(self, x):
        ...


class TLearner(BaseCATELearner):
    """T-learner: the simplest meta-learner, two independent response-surface fits.

    mu0 fit on control units, mu1 fit on treated units, no propensity model at all:
        tau_hat(x) = mu1_hat(x) - mu0_hat(x)
    """

    def __init__(self, base_learner: str = "rf"):
        self.base_learner = base_learner
        self.name = f"TLearner[{base_learner}]"
        self.mu0_model = get_regressor(base_learner)
        self.mu1_model = get_regressor(base_learner)

    def fit(self, x, y, w):
        x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
        self.mu0_model.fit(x[w == 0], y[w == 0])
        self.mu1_model.fit(x[w == 1], y[w == 1])
        return self

    def predict(self, x):
        x = np.asarray(x)
        return self.mu1_model.predict(x) - self.mu0_model.predict(x)
