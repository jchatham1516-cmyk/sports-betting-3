"""MLB feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def ensure_mlb_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "elo_diff": 0.0,
        "starter_rating_home": 0.0,
        "starter_rating_away": 0.0,
        "bullpen_rating_home": 0.0,
        "bullpen_rating_away": 0.0,
        "hitting_rating_home": 0.0,
        "hitting_rating_away": 0.0,
        "home_split_home": 0.0,
        "home_split_away": 0.0,
        "recent_form_home": 0.0,
        "recent_form_away": 0.0,
        "injury_impact_home": 0.0,
        "injury_impact_away": 0.0,
        "rest_days_home": 0.0,
        "rest_days_away": 0.0,
        "travel_distance_home": 0.0,
        "travel_distance_away": 0.0,
        "home_moneyline": 0.0,
        "spread": 0.0,
        "implied_home_prob": 0.5,
        "spread_abs": 0.0,
        "is_favorite": 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def build_mlb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_mlb_core_columns(df)
    df = df.copy()

    df["starter_rating_diff"] = df["starter_rating_home"] - df["starter_rating_away"]
    df["bullpen_rating_diff"] = df["bullpen_rating_home"] - df["bullpen_rating_away"]
    df["hitting_rating_diff"] = df["hitting_rating_home"] - df["hitting_rating_away"]
    df["home_split_diff"] = df["home_split_home"] - df["home_split_away"]
    df["recent_form_diff"] = df["recent_form_home"] - df["recent_form_away"]
    df["injury_impact_diff"] = df["injury_impact_home"] - df["injury_impact_away"]
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["travel_distance"] = df["travel_distance_home"].fillna(0) + df["travel_distance_away"].fillna(0)
    df["travel_fatigue_diff"] = df["travel_distance_home"].fillna(0) - df["travel_distance_away"].fillna(0)

    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="coerce").fillna(0.0)
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
    df["implied_home_prob"] = pd.to_numeric(df["implied_home_prob"], errors="coerce").fillna(0.5)
    df["spread_abs"] = pd.to_numeric(df["spread_abs"], errors="coerce").fillna(df["spread"].abs())
    df["is_favorite"] = pd.to_numeric(df["is_favorite"], errors="coerce").fillna((df["home_moneyline"] < 0).astype(int)).astype(int)
    df["elo_diff"] = pd.to_numeric(df["elo_diff"], errors="coerce").fillna(0.0)
    return df
