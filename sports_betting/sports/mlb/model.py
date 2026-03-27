"""MLB runtime model helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel

from .features import MLB_FEATURE_COLUMNS, build_mlb_features


MIN_TRAINING_ROWS = 20


class MLBModel(DisciplinedBaselineModel):
    sport = "mlb"

    WIN_FEATURES = list(MLB_FEATURE_COLUMNS)


def train_mlb_model(historical_df: pd.DataFrame):
    frame = build_mlb_features(historical_df)
    if len(frame) < MIN_TRAINING_ROWS:
        return None

    y = pd.to_numeric(frame.get("home_win"), errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None

    x = frame.reindex(columns=MLB_FEATURE_COLUMNS, fill_value=0.0).astype(float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_scaled, y)
    model.feature_columns = list(MLB_FEATURE_COLUMNS)
    return model, scaler


def predict_home_probability(model_bundle, games_df: pd.DataFrame) -> pd.Series:
    frame = build_mlb_features(games_df)
    model, scaler = model_bundle
    feature_columns = getattr(model, "feature_columns", MLB_FEATURE_COLUMNS)
    x = frame.reindex(columns=feature_columns, fill_value=0.0).astype(float)
    x_scaled = scaler.transform(x)
    probs = model.predict_proba(x_scaled)[:, 1]
    return pd.Series(probs, index=frame.index, dtype=float)
