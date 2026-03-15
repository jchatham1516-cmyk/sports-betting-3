"""Train NBA historical models with time-aware validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sports_betting.sports.nba.dataset import HISTORICAL_OUTPUT_PATH, build_nba_historical_dataset
from sports_betting.sports.nba.features import build_nba_features
from sports_betting.sports.nba.training import save_nba_models, train_nba_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA moneyline/spread/totals models.")
    parser.add_argument(
        "--historical-csv",
        default=str(HISTORICAL_OUTPUT_PATH),
        help="Path to nba_historical.csv. If missing, dataset builder is run first.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist local model pickles under sports_betting/data/models/.",
    )
    parser.add_argument(
        "--models-dir",
        default="sports_betting/data/models",
        help="Directory used when --save-models is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_path = Path(args.historical_csv)

    if not historical_path.exists():
        historical_path = build_nba_historical_dataset(output_path=historical_path)
        print(f"[NBA] built historical dataset -> {historical_path}")

    historical_df = pd.read_csv(historical_path)
    feature_df = build_nba_features(historical_df)
    # Keep original targets/id fields but use cleaned feature values.
    for column in feature_df.columns:
        if column in historical_df.columns:
            historical_df[column] = feature_df[column]

    artifact = train_nba_models(historical_df)

    print("[NBA] training completed")
    for market, market_metrics in artifact.metrics.items():
        print(f"\n[{market.upper()}]")
        for key, value in market_metrics.items():
            print(f"{key}: {value}")
        if market in artifact.feature_importance:
            top_features = sorted(
                artifact.feature_importance[market].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            print("top_feature_importance:")
            for feature_name, score in top_features:
                print(f"  - {feature_name}: {score:.6f}")

    if args.save_models:
        saved = save_nba_models(artifact, Path(args.models_dir))
        for market, model_path in saved.items():
            print(f"[NBA] saved {market} model -> {model_path}")


if __name__ == "__main__":
    main()
