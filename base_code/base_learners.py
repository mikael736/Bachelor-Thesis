from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import config


def get_regressor(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestRegressor(random_state=random_state)
    if method == "linear":
        return LinearRegression()
    raise ValueError(f"Unknown regressor method '{method}'. Available: 'rf', 'linear'.")


def get_classifier(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier method '{method}'. Available: 'rf'.")
