"""NHL disciplined baseline model."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel


class NHLModel(DisciplinedBaselineModel):
    sport = "nhl"
    PROBABILITY_BOUNDS = {
        "moneyline": (0.14, 0.86),
        "spread": (0.20, 0.80),
        "total": (0.20, 0.80),
    }


NHL_RUNTIME_FEATURES = [
    "goalie_diff",
    "xgf_diff",
    "special_teams_diff",
    "rest_diff",
    "injury_impact_diff",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_nhl_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["goalie_diff"] = _num(out, "goalie_strength_home") - _num(out, "goalie_strength_away")
    out["xgf_diff"] = _num(out, "xgf_home") - _num(out, "xgf_away")
    out["special_teams_diff"] = _num(out, "pp_home") - _num(out, "pk_away")
    out["rest_diff"] = _num(out, "rest_home") - _num(out, "rest_away")
    if "injury_impact_diff" not in out.columns:
        out["injury_impact_diff"] = _num(out, "injury_impact_home") - _num(out, "injury_impact_away")
    else:
        out["injury_impact_diff"] = _num(out, "injury_impact_diff")
    return out


def train_nhl_runtime_model(historical_df: pd.DataFrame):
    frame = build_nhl_features(historical_df)
    if frame.empty:
        raise ValueError("[NHL] Missing historical data - cannot train model")
    y = _num(frame, "home_win").round().clip(0, 1).astype(int)
    if y.nunique() < 2:
        raise ValueError("[NHL] Missing historical data - cannot train model")
    x = frame.reindex(columns=NHL_RUNTIME_FEATURES, fill_value=0.0).astype(float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_scaled, y)
    model.feature_columns = list(NHL_RUNTIME_FEATURES)
    return model, scaler
