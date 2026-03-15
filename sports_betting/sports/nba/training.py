"""Training architecture for NBA moneyline, spread, and totals models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from sports_betting.sports.nba.features import NBA_FEATURE_COLUMNS, build_nba_features


@dataclass
class NBATrainingArtifact:
    """Container for trained models and diagnostics."""

    models: dict[str, CalibratedClassifierCV]
    metrics: dict[str, dict[str, float | int | str]]
    feature_importance: dict[str, dict[str, float]]


MODEL_TARGETS = {
    "moneyline": "home_win",
    "spread": "home_cover",
    "totals": "over_hit",
}


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time (season/date), keeping the newest season for validation when possible."""
    ordered = df.copy()
    ordered["date"] = pd.to_datetime(ordered.get("date"), errors="coerce")
    ordered = ordered.sort_values(["season", "date"]).reset_index(drop=True)

    unique_seasons = [s for s in ordered["season"].dropna().unique().tolist()]
    if len(unique_seasons) >= 2:
        val_season = max(unique_seasons)
        train_df = ordered[ordered["season"] < val_season]
        val_df = ordered[ordered["season"] == val_season]
        if not train_df.empty and not val_df.empty:
            return train_df, val_df

    split_idx = int(len(ordered) * 0.8)
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered.iloc[:split_idx], ordered.iloc[split_idx:]


def _base_estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=0.05,
        max_iter=320,
        max_leaf_nodes=31,
        min_samples_leaf=20,
    )


def _safe_binary_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, 0), errors="coerce").fillna(0).round().clip(0, 1).astype(int)


def _evaluate(y_true: pd.Series, p_hat: pd.Series) -> dict[str, float]:
    y_pred = (p_hat >= 0.5).astype(int)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_hat)) if y_true.nunique() > 1 else 0.5,
        "brier": float(brier_score_loss(y_true, p_hat)),
        "log_loss": float(log_loss(y_true, p_hat, labels=[0, 1])),
    }


def train_nba_models(historical_df: pd.DataFrame) -> NBATrainingArtifact:
    """Train calibrated NBA models using a strict time-based validation split."""
    if historical_df.empty:
        raise ValueError("Cannot train NBA models on an empty dataframe.")

    data = build_nba_features(historical_df)
    if "season" not in data.columns:
        data["season"] = pd.to_datetime(data.get("date"), errors="coerce").dt.year.fillna(0).astype(int)

    train_df, val_df = _time_split(data)
    if train_df.empty or val_df.empty:
        raise ValueError("Time-based split failed; need both train and validation rows.")

    models: dict[str, CalibratedClassifierCV] = {}
    metrics: dict[str, dict[str, float | int | str]] = {}
    importances: dict[str, dict[str, float]] = {}

    for market, target in MODEL_TARGETS.items():
        y_train = _safe_binary_series(train_df, target)
        y_val = _safe_binary_series(val_df, target)

        # Handle degenerate training targets gracefully.
        if y_train.nunique() < 2:
            metrics[market] = {
                "rows_train": int(len(train_df)),
                "rows_validation": int(len(val_df)),
                "note": "single-class training target; skipped model fit",
                "base_rate": float(y_train.mean()) if len(y_train) else 0.0,
            }
            continue

        x_train = train_df[NBA_FEATURE_COLUMNS]
        x_val = val_df[NBA_FEATURE_COLUMNS]

        base_model = _base_estimator()
        base_model.fit(x_train, y_train)

        calibrator = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
        calibrator.fit(x_val, y_val)

        p_val = pd.Series(calibrator.predict_proba(x_val)[:, 1], index=x_val.index).clip(0.01, 0.99)
        market_metrics = _evaluate(y_val, p_val)
        market_metrics.update({
            "rows_train": int(len(train_df)),
            "rows_validation": int(len(val_df)),
            "train_seasons": str(sorted(train_df["season"].dropna().unique().tolist())),
            "validation_seasons": str(sorted(val_df["season"].dropna().unique().tolist())),
        })

        perm = permutation_importance(base_model, x_val, y_val, n_repeats=5, random_state=42, n_jobs=1)
        importances[market] = {
            col: float(score) for col, score in zip(NBA_FEATURE_COLUMNS, perm.importances_mean, strict=False)
        }

        models[market] = calibrator
        metrics[market] = market_metrics

    return NBATrainingArtifact(models=models, metrics=metrics, feature_importance=importances)


def save_nba_models(artifact: NBATrainingArtifact, output_dir: Path) -> dict[str, Path]:
    """Persist trained models to local `data/models` directory (gitignored by policy)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for market, model in artifact.models.items():
        path = output_dir / f"nba_{market}_model.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        saved_paths[market] = path

    return saved_paths


def load_nba_models(model_dir: Path) -> dict[str, CalibratedClassifierCV]:
    """Load local NBA models when available."""
    loaded: dict[str, CalibratedClassifierCV] = {}
    for market in MODEL_TARGETS:
        path = model_dir / f"nba_{market}_model.pkl"
        if path.exists():
            with path.open("rb") as handle:
                loaded[market] = pickle.load(handle)
    return loaded
