"""MLB feature engineering helpers."""

from __future__ import annotations

import pandas as pd

MLB_SOURCE_COLUMNS = [
    "pitcher_whip_home",
    "pitcher_whip_away",
    "pitcher_k_rate_home",
    "pitcher_k_rate_away",
    "pitcher_era_home",
    "pitcher_era_away",
    "starter_rating_home",
    "starter_rating_away",
    "bullpen_rating_home",
    "bullpen_rating_away",
    "hitting_rating_home",
    "hitting_rating_away",
    "home_split_home",
    "home_split_away",
    "recent_form_home",
    "recent_form_away",
]

MLB_REQUIRED_SOURCE_COLUMNS = [
    "starter_rating_home",
    "starter_rating_away",
    "bullpen_rating_home",
    "bullpen_rating_away",
    "hitting_rating_home",
    "hitting_rating_away",
]


def _num(df: pd.DataFrame, col: str, default: float = float("nan")) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def enrich_mlb_live_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    missing = [col for col in MLB_REQUIRED_SOURCE_COLUMNS if col not in out.columns]
    if missing:
        print(f"[MLB WARNING] Missing advanced stats: {missing} — using fallback features")

    if "home_split_home" not in out.columns:
        out["home_split_home"] = pd.to_numeric(out.get("runs_per_game_home"), errors="coerce")
    if "home_split_away" not in out.columns:
        out["home_split_away"] = pd.to_numeric(out.get("runs_per_game_away"), errors="coerce")

    if "recent_form_home" not in out.columns:
        out["recent_form_home"] = pd.to_numeric(out.get("pitcher_last3_starts_home"), errors="coerce")
    if "recent_form_away" not in out.columns:
        out["recent_form_away"] = pd.to_numeric(out.get("pitcher_last3_starts_away"), errors="coerce")

    for col in MLB_SOURCE_COLUMNS:
        out[col] = pd.to_numeric(
            out.get(col, pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)

    print(
        "[MLB SOURCE DEBUG]",
        out[[
            "starter_rating_home",
            "starter_rating_away",
            "bullpen_rating_home",
            "bullpen_rating_away",
            "hitting_rating_home",
            "hitting_rating_away",
        ]].head(),
    )

    return out


def ensure_mlb_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "elo_diff": 0.0,
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
        "pitcher_name_home": "",
        "pitcher_name_away": "",
        "pitcher_era_home": 0.0,
        "pitcher_era_away": 0.0,
        "pitcher_whip_home": 0.0,
        "pitcher_whip_away": 0.0,
        "pitcher_k_rate_home": 0.0,
        "pitcher_k_rate_away": 0.0,
        "pitcher_diff": 0.0,
        "pitcher_whip_diff": 0.0,
        "pitcher_k_rate_diff": 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def build_mlb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_mlb_core_columns(df)
    df = df.copy()
    df["pitcher_era_home"] = _num(df, "pitcher_era_home", default=4.20).replace(0, 4.20).fillna(4.20)
    df["pitcher_era_away"] = _num(df, "pitcher_era_away", default=4.20).replace(0, 4.20).fillna(4.20)
    df["pitcher_whip_home"] = _num(df, "pitcher_whip_home", default=1.28).replace(0, 1.28).fillna(1.28)
    df["pitcher_whip_away"] = _num(df, "pitcher_whip_away", default=1.28).replace(0, 1.28).fillna(1.28)
    df["pitcher_k_rate_home"] = _num(df, "pitcher_k_rate_home", default=0.22).replace(0, 0.22).fillna(0.22)
    df["pitcher_k_rate_away"] = _num(df, "pitcher_k_rate_away", default=0.22).replace(0, 0.22).fillna(0.22)

    if {"pitcher_era_home", "pitcher_era_away"}.issubset(df.columns):
        df["pitcher_diff"] = df["pitcher_era_away"] - df["pitcher_era_home"]
    else:
        df["pitcher_diff"] = _num(df, "pitcher_diff", default=0.0)
    df["pitcher_whip_diff"] = df["pitcher_whip_away"] - df["pitcher_whip_home"]
    df["pitcher_k_rate_diff"] = df["pitcher_k_rate_home"] - df["pitcher_k_rate_away"]

    if {"starter_rating_home", "starter_rating_away"}.issubset(df.columns):
        df["starter_rating_diff"] = _num(df, "starter_rating_home") - _num(df, "starter_rating_away")
    else:
        df["starter_rating_diff"] = _num(df, "elo_diff")

    if {"bullpen_rating_home", "bullpen_rating_away"}.issubset(df.columns):
        df["bullpen_rating_diff"] = _num(df, "bullpen_rating_home") - _num(df, "bullpen_rating_away")
    else:
        df["bullpen_rating_diff"] = _num(df, "recent_form_home") - _num(df, "recent_form_away")

    if {"hitting_rating_home", "hitting_rating_away"}.issubset(df.columns):
        df["hitting_rating_diff"] = _num(df, "hitting_rating_home") - _num(df, "hitting_rating_away")
    else:
        runs_diff = _num(df, "runs_per_game_home") - _num(df, "runs_per_game_away")
        recent_form_diff = _num(df, "recent_form_home") - _num(df, "recent_form_away")
        df["hitting_rating_diff"] = runs_diff.combine_first(recent_form_diff)

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
