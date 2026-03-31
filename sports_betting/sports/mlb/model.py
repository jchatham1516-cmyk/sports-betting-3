"""MLB runtime model helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel
from sports_betting.sports.mlb.features import build_mlb_features
from sports_betting.sports.mlb.schema import MLB_REQUIRED_FEATURES


LOGGER = logging.getLogger(__name__)
MIN_TRAINING_ROWS = 20


@dataclass
class MLBModelBundle:
    runtime_model: object
    feature_columns: list[str]


class MLBModel(DisciplinedBaselineModel):
    sport = "mlb"
    WIN_FEATURES = list(MLB_REQUIRED_FEATURES)


def train_mlb_model(historical_df: pd.DataFrame) -> MLBModelBundle:
    df = build_mlb_features(historical_df).copy()
    df = df.dropna(subset=["home_win"])
    if len(df) < MIN_TRAINING_ROWS:
        raise ValueError(f"[MLB] Insufficient historical rows ({len(df)}) for runtime training; need at least {MIN_TRAINING_ROWS}.")

    X = df[MLB_REQUIRED_FEATURES].fillna(0.0)
    feature_columns = X.columns.tolist()
    y = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError("[MLB] Historical training target requires two classes in home_win.")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
    )
    model.fit(X, y)
    return MLBModelBundle(runtime_model=model, feature_columns=feature_columns)


def predict_mlb_model(bundle: MLBModelBundle, daily_df: pd.DataFrame) -> pd.DataFrame:
    df = build_mlb_features(daily_df).copy()

    X = df.reindex(columns=bundle.feature_columns, fill_value=0.0).fillna(0.0)
    probs = bundle.runtime_model.predict_proba(X)[:, 1]
    df["predicted_home_win_prob"] = probs
    df["predicted_away_win_prob"] = 1.0 - probs
    return df


def save_mlb_model_bundle(bundle: MLBModelBundle, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return model_path


def load_mlb_model_bundle(model_path: Path) -> MLBModelBundle:
    loaded = joblib.load(model_path)
    if isinstance(loaded, MLBModelBundle):
        return loaded
    if isinstance(loaded, tuple) and len(loaded) >= 1:
        runtime_model = loaded[0]
        return MLBModelBundle(runtime_model=runtime_model, feature_columns=list(MLB_REQUIRED_FEATURES))
    LOGGER.warning("[MLB] Loaded artifact had unexpected shape; inferring feature columns.")
    return MLBModelBundle(runtime_model=loaded, feature_columns=list(MLB_REQUIRED_FEATURES))
