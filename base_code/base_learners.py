import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

# MLPs are sensitive to feature scale and to under-fitting if capped at sklearn's default
# max_iter=200, so wrap them in a scaling pipeline and give them more iterations to converge.
_MLP_KWARGS = dict(hidden_layer_sizes=(32, 16), max_iter=2000)


class ExponentialRegressor:
    """f(x) = a * exp(b * x): a 2-parameter exponential curve fit by nonlinear least squares -
    the correctly-specified family for response_surface.exponential, analogous to how
    LinearRegression is the correctly-specified family for response_surface.linear. Fits directly
    in y-space (not via log(y)), so it works even where y is negative (e.g. under additive noise).
    """

    @staticmethod
    def _model(x, a, b):
        return a * np.exp(b * x)

    def fit(self, x, y):
        x = np.asarray(x)[:, 0]
        self.params_, _ = curve_fit(self._model, x, np.asarray(y), p0=[1.0, 1.0], maxfev=10000)
        return self

    def predict(self, x):
        x = np.asarray(x)[:, 0]
        return self._model(x, *self.params_)


def get_regressor(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestRegressor(random_state=random_state)
    if method == "linear":
        return LinearRegression()
    if method == "exponential":
        return ExponentialRegressor()
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPRegressor(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown regressor method '{method}'. Available: 'rf', 'linear', 'exponential', 'nn'.")


def get_classifier(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestClassifier(random_state=random_state)
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown classifier method '{method}'. Available: 'rf', 'nn'.")
