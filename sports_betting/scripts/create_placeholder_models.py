"""Create fitted placeholder model artifacts at workflow runtime only."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from sports_betting.sports.nba.model import EnsembleProbabilityCalibrator, NBAModel
from sports_betting.sports.nfl.model import NFLModel
from sports_betting.sports.nhl.model import NHLModel

MODEL_DIR = Path("sports_betting/data/models")
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
    model.fit(x, y)
    return model


def build_payload(sport: str, model_cls: type[NBAModel]) -> dict:
    """Create payload keys expected by load_artifact() and prediction paths."""
    win_features = list(model_cls.WIN_FEATURES)
    spread_features = list(model_cls.SPREAD_FEATURES)
    total_features = list(model_cls.TOTAL_FEATURES)

    return {
        "sport": sport,
        "win_features": win_features,
        "spread_features": spread_features,
        "total_features": total_features,
        "moneyline_models": [fitted_model(len(win_features))],
        "spread_models": [fitted_model(len(spread_features))],
        "total_models": [fitted_model(len(total_features))],
        "moneyline_cal": EnsembleProbabilityCalibrator(),
        "spread_cal": EnsembleProbabilityCalibrator(),
        "total_cal": EnsembleProbabilityCalibrator(),
        "metrics": {},
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for sport, model_cls in MODEL_CLASSES.items():
        path = MODEL_DIR / f"{sport}_model.pkl"
        with path.open("wb") as file_obj:
            pickle.dump(build_payload(sport, model_cls), file_obj)

    print("Placeholder model artifacts created successfully at runtime.")


if __name__ == "__main__":
    main()
