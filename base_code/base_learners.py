from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import config


def get_regressor(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestRegressor(random_state=random_state)
    raise ValueError(f"Unknown regressor method '{method}'. Available: 'rf'.")


def get_classifier(method: str = "rf", *, random_state: int = config.SEED):
    if method == "rf":
        return RandomForestClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier method '{method}'. Available: 'rf'.")
