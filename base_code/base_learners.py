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

def get_regressor(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestRegressor(random_state=random_state)
    if method == "linear":
        return LinearRegression()
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPRegressor(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown regressor method '{method}'. Available: 'rf', 'linear', 'nn'.")


def get_classifier(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestClassifier(random_state=random_state)
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown classifier method '{method}'. Available: 'rf', 'nn'.")
