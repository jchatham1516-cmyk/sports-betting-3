"""Train and persist sport model artifacts from historical CSV inputs."""

from __future__ import annotations

import argparse

import pandas as pd

from main import choose_model
from sports_betting.scripts.data_io import (
    historical_file_path,
    model_artifact_path,
    required_historical_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train saved model artifacts for one or more sports.")
    parser.add_argument(
        "--sports",
        nargs="+",
        choices=["nba", "nfl", "nhl"],
        default=["nba", "nfl", "nhl"],
        help="Sports to train.",
    )
    return parser.parse_args()


def _validate_historical_schema(df: pd.DataFrame, sport: str) -> None:
    required = required_historical_columns(sport)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{sport.upper()}] Missing required historical columns: {', '.join(missing)}. "
            f"Expected columns include: {', '.join(required)}"
        )


def train_sport_model(sport: str) -> None:
    hist_path = historical_file_path(sport)
    artifact_path = model_artifact_path(sport)

    if not hist_path.exists():
        raise RuntimeError(f"[{sport.upper()}] Historical CSV does not exist: {hist_path}")

    historical = pd.read_csv(hist_path)
    _validate_historical_schema(historical, sport)

    model = choose_model(sport)
    model.train(historical)
    model.save_artifact(artifact_path)

    print(f"[{sport.upper()}] Trained model artifact written to: {artifact_path}")
    if getattr(model, "metrics", None):
        print(f"[{sport.upper()}] Training metrics: {model.metrics}")


def main() -> None:
    args = parse_args()
    for sport in args.sports:
        train_sport_model(sport)


if __name__ == "__main__":
    main()
