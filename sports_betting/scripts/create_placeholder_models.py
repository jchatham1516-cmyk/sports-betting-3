"""Create minimal placeholder model artifacts for CI/workflow runs."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("sports_betting/data/models")
SPORTS = ("nba", "nfl", "nhl")


def build_payload(sport: str) -> dict:
    return {
        "sport": sport,
        "win_features": [],
        "spread_features": [],
        "total_features": [],
        "moneyline_models": [LogisticRegression()],
        "spread_models": [LogisticRegression()],
        "total_models": [LogisticRegression()],
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
