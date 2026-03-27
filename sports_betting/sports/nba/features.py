"""Feature engineering utilities for NBA training and daily inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

NBA_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "travel_fatigue_diff",
    "injury_impact_diff",
    "net_rating_diff",
    "pace_diff",
    "last5_net_rating_diff",
    "last10_net_rating_diff",
    "market_prob_home",
]

NBA_SOURCE_COLUMNS = [
    "offensive_rating_home",
    "offensive_rating_away",
    "defensive_rating_home",
    "defensive_rating_away",
    "net_rating_home",
    "net_rating_away",
    "pace_home",
    "pace_away",
    "true_shooting_home",
    "true_shooting_away",
    "effective_fg_home",
    "effective_fg_away",
    "turnover_rate_home",
    "turnover_rate_away",
    "rebound_rate_home",
    "rebound_rate_away",
    "free_throw_rate_home",
    "free_throw_rate_away",
]

NBA_REQUIRED_SOURCE_COLUMNS = [
    "offensive_rating_home",
    "offensive_rating_away",
    "defensive_rating_home",
    "defensive_rating_away",
    "net_rating_home",
    "net_rating_away",
    "true_shooting_home",
    "true_shooting_away",
    "effective_fg_home",
    "effective_fg_away",
    "turnover_rate_home",
    "turnover_rate_away",
    "rebound_rate_home",
    "rebound_rate_away",
    "free_throw_rate_home",
    "free_throw_rate_away",
]

# Conservative hard caps to keep outliers from dominating tree splits.
FEATURE_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "elo_diff": (-450.0, 450.0),
    "rest_diff": (-5.0, 5.0),
    "travel_fatigue_diff": (-6000.0, 6000.0),
    "injury_impact_diff": (-25.0, 25.0),
    "net_rating_diff": (-35.0, 35.0),
    "pace_diff": (-15.0, 15.0),
    "last5_net_rating_diff": (-40.0, 40.0),
    "last10_net_rating_diff": (-30.0, 30.0),
    "market_prob_home": (0.02, 0.98),
}


def _coalesce_numeric(df: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    """Return first present numeric column from `columns` or a default series."""
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _american_to_prob(odds: pd.Series) -> pd.Series:
    """Convert American odds to implied probability with stable handling for malformed values."""
    numeric_odds = pd.to_numeric(odds, errors="coerce")
    prob = np.where(
        numeric_odds < 0,
        (-numeric_odds) / ((-numeric_odds) + 100.0),
        np.where(numeric_odds > 0, 100.0 / (numeric_odds + 100.0), np.nan),
    )
    return pd.Series(prob, index=odds.index, dtype="float64")


def _merge_nba_team_stats(df: pd.DataFrame, nba_team_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stats = nba_team_stats.copy()
    if "team" not in stats.columns:
        raise RuntimeError("[DATA ERROR] nba_team_stats is missing required column: team")

    if "home_team_norm" not in out.columns or "away_team_norm" not in out.columns:
        if "home_team" not in out.columns or "away_team" not in out.columns:
            raise RuntimeError("[DATA ERROR] Missing team columns required for NBA stat merge")
        out["home_team_norm"] = out["home_team"].astype(str).str.lower().str.strip()
        out["away_team_norm"] = out["away_team"].astype(str).str.lower().str.strip()

    stats["team"] = stats["team"].astype(str).str.lower().str.strip()
    out = out.merge(
        stats,
        left_on="home_team_norm",
        right_on="team",
        how="left",
    )
    out = out.merge(
        stats,
        left_on="away_team_norm",
        right_on="team",
        how="left",
        suffixes=("_home", "_away"),
    )
    return out


def enrich_nba_live_features(df: pd.DataFrame, nba_team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()

    if nba_team_stats is not None:
        out = _merge_nba_team_stats(out, nba_team_stats)

    alias_map: dict[str, list[str]] = {
        "offensive_rating_home": ["offensive_rating_home", "off_rating_home"],
        "offensive_rating_away": ["offensive_rating_away", "off_rating_away"],
        "defensive_rating_home": ["defensive_rating_home", "def_rating_home"],
        "defensive_rating_away": ["defensive_rating_away", "def_rating_away"],
        "net_rating_home": ["net_rating_home"],
        "net_rating_away": ["net_rating_away"],
        "pace_home": ["pace_home"],
        "pace_away": ["pace_away"],
        "true_shooting_home": ["true_shooting_home"],
        "true_shooting_away": ["true_shooting_away"],
        "effective_fg_home": ["effective_fg_home", "efg_home"],
        "effective_fg_away": ["effective_fg_away", "efg_away"],
        "turnover_rate_home": ["turnover_rate_home"],
        "turnover_rate_away": ["turnover_rate_away"],
        "rebound_rate_home": ["rebound_rate_home"],
        "rebound_rate_away": ["rebound_rate_away"],
        "free_throw_rate_home": ["free_throw_rate_home"],
        "free_throw_rate_away": ["free_throw_rate_away"],
    }

    for target, candidates in alias_map.items():
        if target not in out.columns:
            out[target] = _coalesce_numeric(out, candidates, default=np.nan)
        out[target] = pd.to_numeric(out[target], errors="coerce")

    print(
        "[NBA SOURCE DEBUG]",
        out[[
            "offensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_home",
            "defensive_rating_away",
        ]].head(),
    )

    missing = [col for col in NBA_REQUIRED_SOURCE_COLUMNS if col not in out.columns]
    if missing:
        raise RuntimeError(f"[DATA ERROR] Missing required source stats: {missing}")

    for col in NBA_SOURCE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def build_nba_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["offensive_rating_diff"] = out["offensive_rating_home"] - out["offensive_rating_away"]
    out["defensive_rating_diff"] = out["defensive_rating_home"] - out["defensive_rating_away"]
    out["net_rating_diff"] = out["net_rating_home"] - out["net_rating_away"]
    out["pace_diff"] = out["pace_home"] - out["pace_away"]
    out["true_shooting_diff"] = out["true_shooting_home"] - out["true_shooting_away"]
    out["effective_fg_diff"] = out["effective_fg_home"] - out["effective_fg_away"]
    out["turnover_rate_diff"] = out["turnover_rate_home"] - out["turnover_rate_away"]
    out["rebound_rate_diff"] = out["rebound_rate_home"] - out["rebound_rate_away"]
    out["free_throw_rate_diff"] = out["free_throw_rate_home"] - out["free_throw_rate_away"]
    return out


def build_nba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic NBA model features for training and daily prediction.

    The function is intentionally idempotent and side-effect free: it returns a copy,
    computes missing diffs from home/away columns when needed, fills nulls using
    training-safe defaults, and clips heavy outliers.
    """

    out = df.copy()

    out["elo_diff"] = _coalesce_numeric(out, ["elo_diff", "elo_home", "elo_away"], default=0.0)
    if "elo_home" in out.columns and "elo_away" in out.columns:
        out["elo_diff"] = pd.to_numeric(out["elo_home"], errors="coerce") - pd.to_numeric(out["elo_away"], errors="coerce")

    out["rest_diff"] = _coalesce_numeric(out, ["rest_diff"], default=np.nan)
    if out["rest_diff"].isna().all():
        out["rest_diff"] = _coalesce_numeric(out, ["rest_days_home"], 0.0) - _coalesce_numeric(out, ["rest_days_away"], 0.0)

    out["travel_fatigue_diff"] = _coalesce_numeric(out, ["travel_fatigue_diff"], default=np.nan)
    if out["travel_fatigue_diff"].isna().all():
        away_fatigue = (
            _coalesce_numeric(out, ["travel_distance_away"], 0.0)
            + 250.0 * _coalesce_numeric(out, ["timezone_shift_away"], 0.0)
            + 150.0 * _coalesce_numeric(out, ["road_trip_length_away"], 0.0)
        )
        home_fatigue = (
            _coalesce_numeric(out, ["travel_distance_home"], 0.0)
            + 250.0 * _coalesce_numeric(out, ["timezone_shift_home"], 0.0)
            + 150.0 * _coalesce_numeric(out, ["road_trip_length_home"], 0.0)
        )
        out["travel_fatigue_diff"] = away_fatigue - home_fatigue

    out["injury_impact_diff"] = _coalesce_numeric(out, ["injury_impact_diff"], default=np.nan)
    if out["injury_impact_diff"].isna().all():
        out["injury_impact_diff"] = _coalesce_numeric(out, ["injury_impact_home"], 0.0) - _coalesce_numeric(out, ["injury_impact_away"], 0.0)

    out["net_rating_diff"] = _coalesce_numeric(out, ["net_rating_diff"], default=np.nan)
    if out["net_rating_diff"].isna().all():
        out["net_rating_diff"] = _coalesce_numeric(out, ["net_rating_home"], 0.0) - _coalesce_numeric(out, ["net_rating_away"], 0.0)

    out["pace_diff"] = _coalesce_numeric(out, ["pace_diff"], default=np.nan)
    if out["pace_diff"].isna().all():
        out["pace_diff"] = _coalesce_numeric(out, ["pace_home"], 0.0) - _coalesce_numeric(out, ["pace_away"], 0.0)

    out["last5_net_rating_diff"] = _coalesce_numeric(out, ["last5_net_rating_diff"], default=np.nan)
    if out["last5_net_rating_diff"].isna().all():
        out["last5_net_rating_diff"] = _coalesce_numeric(out, ["last5_net_rating_home"], 0.0) - _coalesce_numeric(out, ["last5_net_rating_away"], 0.0)

    out["last10_net_rating_diff"] = _coalesce_numeric(out, ["last10_net_rating_diff"], default=np.nan)
    if out["last10_net_rating_diff"].isna().all():
        out["last10_net_rating_diff"] = _coalesce_numeric(out, ["last10_net_rating_home"], 0.0) - _coalesce_numeric(out, ["last10_net_rating_away"], 0.0)

    out["market_prob_home"] = _coalesce_numeric(out, ["market_prob_home"], default=np.nan)
    if out["market_prob_home"].isna().all():
        if "closing_moneyline_home" in out.columns:
            out["market_prob_home"] = _american_to_prob(out["closing_moneyline_home"])
        elif "home_odds" in out.columns:
            out["market_prob_home"] = _american_to_prob(out["home_odds"])

    for column in NBA_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if column == "market_prob_home":
            out[column] = out[column].fillna(0.5)
        else:
            out[column] = out[column].fillna(0.0)

        lo, hi = FEATURE_CLIP_BOUNDS[column]
        out[column] = out[column].clip(lo, hi)

    return out
