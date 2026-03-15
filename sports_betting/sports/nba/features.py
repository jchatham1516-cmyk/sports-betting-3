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
