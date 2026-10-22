"""Base learners for the meta-learners: random forest (default), linear regression and neural network."""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# MLP settings: standardized inputs, max_iter raised from sklearn's default
_MLP_KWARGS = dict(hidden_layer_sizes=(32, 16), max_iter=2000)


def get_regressor(method: str = "rf", *, random_state: int, rf_params: dict | None = None):
    """rf_params are passed to the random forest (e.g. min_samples_leaf); set in the running script."""
    if method == "rf":
        return RandomForestRegressor(random_state=random_state, **(rf_params or {}))
    if method == "linear":
        return LinearRegression()
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPRegressor(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown regressor method '{method}'. Available: 'rf', 'linear', 'nn'.")


def get_classifier(method: str = "rf", *, random_state: int, rf_params: dict | None = None):
    """rf_params are passed to the random forest (e.g. min_samples_leaf); set in the running script."""
    if method == "rf":
        return RandomForestClassifier(random_state=random_state, **(rf_params or {}))
    if method == "nn":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state, **_MLP_KWARGS))
    raise ValueError(f"Unknown classifier method '{method}'. Available: 'rf', 'nn'.")
