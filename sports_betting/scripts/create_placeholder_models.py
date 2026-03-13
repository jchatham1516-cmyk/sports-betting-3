"""Create minimal fitted placeholder model artifacts for CI/workflow runs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("sports_betting/data/models")
SPORTS = ("nba", "nfl", "nhl")


class IdentityCalibrator:
    """No-op probability calibrator used by placeholder artifacts."""

    def __init__(self, lower: float = 1e-6, upper: float = 1 - 1e-6) -> None:
        self.lower = lower
        self.upper = upper

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probs, dtype=float), self.lower, self.upper)


def fitted_model() -> LogisticRegression:
    """Build a tiny fitted binary classifier safe for predict_proba()."""
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
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
        "moneyline_cal": IdentityCalibrator(),
        "spread_cal": IdentityCalibrator(),
        "total_cal": IdentityCalibrator(),
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
