"""Preflight validation checks for production runs."""

from __future__ import annotations

import argparse

from sports_betting.scripts.data_io import validate_historical_requirements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate required historical/model inputs before running pipeline.")
    parser.add_argument(
        "--sports",
        nargs="+",
        choices=["nba", "nfl", "nhl"],
        default=["nba", "nfl", "nhl"],
        help="Sports to validate.",
    )
    parser.add_argument(
        "--require-historical-csv",
        action="store_true",
        help="Require historical CSV files even if trained model artifacts exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_historical_requirements(
        sports=args.sports,
        allow_model_artifacts=not args.require_historical_csv,
        validate_schema=True,
    )
    print("Preflight validation passed.")


if __name__ == "__main__":
    main()
