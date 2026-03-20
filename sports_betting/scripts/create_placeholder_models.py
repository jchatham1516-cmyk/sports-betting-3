"""Create fitted placeholder model artifacts at workflow runtime only."""

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from sports_betting.sports.nba.model import NBAModel
from sports_betting.sports.nfl.model import NFLModel
from sports_betting.sports.nhl.model import NHLModel

MODEL_DIR = "sports_betting/data/models"
MODEL_CLASSES = {
    "nba": NBAModel,
    "nfl": NFLModel,
    "nhl": NHLModel,
}


def fitted_model(feature_count: int) -> LogisticRegression:
    """Build a tiny fitted binary classifier with a specific feature count."""
    x = np.vstack([np.zeros(feature_count), np.ones(feature_count)])
    y = np.array([0, 1])
    model = LogisticRegression(random_state=42, solver="liblinear")
    # Save feature order if DataFrame
    if hasattr(x, "columns"):
        model.feature_columns = list(x.columns)
        x_train = x.values
    else:
        model.feature_columns = None
        x_train = x
    model.fit(x_train, y)
    return model


def build_payload(sport: str, model_cls: type[NBAModel]) -> dict:
    """Create payload keys expected by load_artifact() and prediction paths."""
    win_feature_count = len(model_cls.WIN_FEATURES)
    spread_feature_count = len(model_cls.SPREAD_FEATURES)
    total_feature_count = len(model_cls.TOTAL_FEATURES)

    return {
        "sport": sport,
        "moneyline_model": fitted_model(win_feature_count),
        "spread_model": fitted_model(spread_feature_count),
        "total_model": fitted_model(total_feature_count),
        "moneyline_cal": None,
        "spread_cal": None,
        "total_cal": None,
        "metrics": {},
        "feature_importance": {},
    }


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    for sport, model_cls in MODEL_CLASSES.items():
        path = os.path.join(MODEL_DIR, f"{sport}_model.pkl")
        with open(path, "wb") as file_obj:
            pickle.dump(build_payload(sport, model_cls), file_obj)

    print("Placeholder model artifacts created successfully at runtime.")


if __name__ == "__main__":
    main()
