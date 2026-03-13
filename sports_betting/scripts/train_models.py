"""Train and persist sport model artifacts from historical CSV inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from main import choose_model
from sports_betting.scripts.build_historical_dataset import build_historical_dataset
from sports_betting.scripts.data_io import (
    load_historical_dataset,
    load_nba_historical_dataset,
    load_nfl_historical_dataset,
    load_nhl_historical_dataset,
    model_artifact_path,
)

SPORT_DATASET_LOADERS = {
    "nba": load_nba_historical_dataset,
    "nfl": load_nfl_historical_dataset,
    "nhl": load_nhl_historical_dataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train saved model artifacts for one or more sports.")
    parser.add_argument(
        "--build-historical",
        action="store_true",
        help="Regenerate enriched historical datasets before training.",
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        choices=["nba", "nfl", "nhl"],
        default=["nba", "nfl", "nhl"],
        help="Sports to train.",
    )
    return parser.parse_args()


def train_sport_model(sport: str, build_historical: bool = False) -> None:
    if build_historical:
        build_historical_dataset(sport, Path("sports_betting/data/raw"))
    artifact_path = model_artifact_path(sport)
    loader = SPORT_DATASET_LOADERS.get(sport, lambda: load_historical_dataset(sport))
    historical = loader()

    model = choose_model(sport)
    model.train(historical)
    model.save_artifact(artifact_path)

    print(f"[{sport.upper()}] Trained model artifact written to: {artifact_path}")
    if getattr(model, "metrics", None):
        print(f"[{sport.upper()}] Training metrics: {model.metrics}")
    if getattr(model, "feature_importance", None):
        print(f"[{sport.upper()}] Feature importance: {model.feature_importance}")


def main() -> None:
    args = parse_args()
    for sport in args.sports:
        train_sport_model(sport, build_historical=args.build_historical)


if __name__ == "__main__":
    main()
