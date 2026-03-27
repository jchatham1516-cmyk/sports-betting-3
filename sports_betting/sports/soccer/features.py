"""Soccer feature engineering helpers."""

from __future__ import annotations

import pandas as pd


SOCCER_FEATURE_COLUMNS = [
    "xg_for_diff",
    "xg_against_diff",
    "recent_form_diff",
    "home_advantage",
    "rest_diff",
    "injury_impact_diff",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_soccer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["xg_for_diff"] = _num(out, "xg_for_home") - _num(out, "xg_for_away")
    out["xg_against_diff"] = _num(out, "xg_against_home") - _num(out, "xg_against_away")
    out["recent_form_diff"] = _num(out, "form_last5_home") - _num(out, "form_last5_away")
    out["home_advantage"] = _num(out, "home_advantage", default=1.0)
    out["rest_diff"] = _num(out, "rest_days_home") - _num(out, "rest_days_away")
    if "injury_impact_diff" not in out.columns:
        out["injury_impact_diff"] = _num(out, "injury_impact_home") - _num(out, "injury_impact_away")
    else:
        out["injury_impact_diff"] = _num(out, "injury_impact_diff")
    return out


def enrich_soccer_live_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Soccer still depends on historical data/model artifacts; this only normalizes live columns.
    for col in [
        "xg_for_home",
        "xg_for_away",
        "xg_against_home",
        "xg_against_away",
        "form_last5_home",
        "form_last5_away",
    ]:
        if col not in out.columns:
            out[col] = 0.0
    return out
