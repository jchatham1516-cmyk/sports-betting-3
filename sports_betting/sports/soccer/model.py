"""Soccer 3-way model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import SOCCER_FEATURE_COLUMNS, build_soccer_features


MIN_TRAINING_ROWS = 30


def _target_outcome(df: pd.DataFrame) -> pd.Series:
    if "match_outcome" in df.columns:
        mapping = {"home": 0, "draw": 1, "away": 2}
        return df["match_outcome"].astype(str).str.lower().map(mapping).fillna(1).astype(int)
    home_score = pd.to_numeric(df.get("home_score"), errors="coerce").fillna(0)
    away_score = pd.to_numeric(df.get("away_score"), errors="coerce").fillna(0)
    out = np.where(home_score > away_score, 0, np.where(home_score < away_score, 2, 1))
    return pd.Series(out, index=df.index, dtype=int)


def train_soccer_model(historical_df: pd.DataFrame):
    frame = build_soccer_features(historical_df)
    if len(frame) < MIN_TRAINING_ROWS:
        return None

    y = _target_outcome(frame)
    if y.nunique() < 3:
        return None

    x = frame.reindex(columns=SOCCER_FEATURE_COLUMNS, fill_value=0.0).astype(float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1200, multi_class="multinomial")
    model.fit(x_scaled, y)
    model.feature_columns = list(SOCCER_FEATURE_COLUMNS)
    return model, scaler


def predict_outcome_probabilities(model_bundle, games_df: pd.DataFrame) -> pd.DataFrame:
    frame = build_soccer_features(games_df)
    model, scaler = model_bundle
    x = frame.reindex(columns=getattr(model, "feature_columns", SOCCER_FEATURE_COLUMNS), fill_value=0.0).astype(float)
    probs = model.predict_proba(scaler.transform(x))

    out = pd.DataFrame(
        {
            "home_prob": probs[:, 0],
            "draw_prob": probs[:, 1],
            "away_prob": probs[:, 2],
        },
        index=frame.index,
    )
    total = out.sum(axis=1).replace(0, 1.0)
    out = out.div(total, axis=0).clip(0.001, 0.998)
    out = out.div(out.sum(axis=1), axis=0)
    return out
