"""Create placeholder model artifacts for CI workflows."""

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression


SPORTS = ("nba", "nfl", "nhl")


def create_payload() -> dict[str, LogisticRegression]:
    """Create placeholder model payload expected by load_artifact()."""
    return {
        "moneyline_models": LogisticRegression(),
        "spread_models": LogisticRegression(),
        "total_models": LogisticRegression(),
    }


def main() -> None:
    models_dir = Path("sports_betting/data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for sport in SPORTS:
        model_path = models_dir / f"{sport}_model.pkl"
        with model_path.open("wb") as artifact_file:
            pickle.dump(create_payload(), artifact_file)

    print(f"Successfully created placeholder model artifacts in {models_dir}")


if __name__ == "__main__":
    main()
