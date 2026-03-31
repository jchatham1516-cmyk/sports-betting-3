"""Train sport models with text-only reporting and optional artifact saving."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from main import choose_model
from sports_betting.scripts.build_historical_dataset import build_historical_dataset
from sports_betting.scripts.build_nba_historical_dataset import build_nba_historical_dataset
from sports_betting.scripts.data_io import (
    load_historical_dataset,
    load_nba_historical_dataset,
    load_nfl_historical_dataset,
    load_nhl_historical_dataset,
    model_artifact_path,
)
from sports_betting.sports.nba.simple_model import train_runtime_model

SAVE_MODEL_ARTIFACTS = False
REPORT_PATH = Path("sports_betting/data/models/nba_model_training_report.txt")

SPORT_DATASET_LOADERS = {
    "nba": load_nba_historical_dataset,
    "nfl": load_nfl_historical_dataset,
    "nhl": load_nhl_historical_dataset,
}

NBA_FEATURE_GROUPS = {
    "moneyline": {
        "target": "home_win",
        "features": [
            "elo_diff",
            "rest_diff",
            "travel_fatigue_diff",
            "injury_impact_diff",
            "net_rating_diff",
            "top_rotation_eff_diff",
            "recent_off_rating_diff",
            "recent_def_rating_diff",
        ],
    },
    "spread": {
        "target": "home_cover",
        "features": [
            "rest_diff",
            "injury_impact_diff",
            "net_rating_diff",
            "top_rotation_eff_diff",
            "travel_fatigue_diff",
            "closing_spread_home",
        ],
    },
    "totals": {
        "target": "over_hit",
        "features": [
            "pace_diff",
            "recent_total_trend",
            "injury_impact_diff",
            "closing_total",
            "true_shooting_diff",
            "turnover_rate_diff",
        ],
    },
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
    parser.add_argument(
        "--save-model-artifacts",
        action="store_true",
        help="If set, persist model artifacts to disk (disabled by default).",
    )
    return parser.parse_args()


def _fit_binary_classifier(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, float | int | str]:
    working = frame.copy()
    for feature in features + [target]:
        if feature not in working.columns:
            working[feature] = 0.0
        working[feature] = pd.to_numeric(working[feature], errors="coerce").fillna(0.0)

    X = working[features]
    feature_columns = X.columns.tolist()
    y = working[target].round().clip(0, 1).astype(int)

    if y.nunique() < 2:
        baseline = float(y.mean()) if len(y) else 0.0
        return {
            "rows": int(len(working)),
            "target_mean": baseline,
            "accuracy": baseline,
            "roc_auc": 0.5,
            "brier": 0.25,
            "note": "single-class target; metrics reflect baseline",
        }

    model = LogisticRegression(max_iter=1000)
    # Save feature order if DataFrame
    if hasattr(X, "columns"):
        model.feature_columns = feature_columns
        X_train = X.values
    else:
        model.feature_columns = None
        X_train = X

    model.fit(X_train, y)

    if hasattr(model, "feature_columns") and model.feature_columns is not None:
        X_pred = working.reindex(columns=model.feature_columns, fill_value=0.0).values
    else:
        X_pred = X.values if hasattr(X, "values") else X

    prob = model.predict_proba(X_pred)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "rows": int(len(working)),
        "target_mean": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def _train_nba_text_report(historical: pd.DataFrame) -> str:
    lines = ["NBA in-memory training report", "=" * 40]
    for market, config in NBA_FEATURE_GROUPS.items():
        metrics = _fit_binary_classifier(historical, config["target"], config["features"])
        lines.append(f"[{market.upper()}] target={config['target']}")
        lines.append(f"features={', '.join(config['features'])}")
        for key, value in metrics.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def train_sport_model(sport: str, build_historical: bool = False, save_model_artifacts: bool = SAVE_MODEL_ARTIFACTS) -> None:
    if build_historical:
        if sport == "nba":
            build_nba_historical_dataset(Path("sports_betting/data/raw"))
        else:
            build_historical_dataset(sport, Path("sports_betting/data/raw"))

    loader = SPORT_DATASET_LOADERS.get(sport, lambda: load_historical_dataset(sport))
    historical = loader()
    if "injury_impact_diff" not in historical.columns:
        historical["injury_impact_diff"] = 0

    if sport == "nba":
        report_text = _train_nba_text_report(historical)
        print(report_text)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report_text, encoding="utf-8")
        print(f"[NBA] wrote training report -> {REPORT_PATH}")

    model = choose_model(sport)
    runtime_model = None
    try:
        runtime_model = train_runtime_model(historical) if sport == "nba" else None
    except Exception:
        runtime_model = None

    if sport != "nba":
        model.train(historical)

    if save_model_artifacts:
        artifact_path = model_artifact_path(sport)
        if sport == "nba":
            if runtime_model is not None:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle(runtime_model, artifact_path)
                print(f"[{sport.upper()}] Runtime model artifact written to: {artifact_path}")
            else:
                print(f"[{sport.upper()}] Runtime model unavailable; falling back to implied probability.")
        else:
            model.save_artifact(artifact_path)
            print(f"[{sport.upper()}] Trained model artifact written to: {artifact_path}")
    else:
        print(f"[{sport.upper()}] SAVE_MODEL_ARTIFACTS=False; skipping binary artifact serialization.")

    if getattr(model, "metrics", None):
        print(f"[{sport.upper()}] Training metrics: {model.metrics}")
    if getattr(model, "feature_importance", None):
        print(f"[{sport.upper()}] Feature importance: {model.feature_importance}")


def main() -> None:
    args = parse_args()
    save_artifacts = SAVE_MODEL_ARTIFACTS or args.save_model_artifacts
    for sport in args.sports:
        train_sport_model(sport, build_historical=args.build_historical, save_model_artifacts=save_artifacts)


if __name__ == "__main__":
    main()
