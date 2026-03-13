"""Create minimal fitted placeholder model artifacts for CI/workflow runs."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("sports_betting/data/models")
SPORTS = ("nba", "nfl", "nhl")


def fitted_model() -> LogisticRegression:
    """Build a tiny fitted binary classifier safe for predict_proba()."""
    x = [[0.0], [1.0]]
    y = [0, 1]
    model = LogisticRegression(random_state=42)
    model.fit(x, y)
    return model


def build_payload(sport: str) -> dict:
    return {
        "sport": sport,
        "win_features": [],
        "spread_features": [],
        "total_features": [],
        "moneyline_models": [fitted_model()],
        "spread_models": [fitted_model()],
        "total_models": [fitted_model()],
        "moneyline_cal": None,
        "spread_cal": None,
        "total_cal": None,
        "metrics": {},
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for sport in SPORTS:
        path = MODEL_DIR / f"{sport}_model.pkl"
        with path.open("wb") as file_obj:
            pickle.dump(build_payload(sport), file_obj)

    print("Placeholder model artifacts created successfully.")


if __name__ == "__main__":
    main()
