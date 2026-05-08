"""Feature engineering utilities for NBA training and daily inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sports_betting.sports.common.team_names import normalize_team_name as shared_normalize_team_name

NBA_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "rest_advantage_weighted",
    "travel_fatigue_diff",
    "injury_impact_diff",
    "net_rating_diff",
    "pace_diff",
    "rolling_off_rating_diff_last5",
    "rolling_def_rating_diff_last5",
    "home_away_net_rating_split_diff",
    "pace_adjusted_scoring_diff",
    "last5_net_rating_diff",
    "last10_net_rating_diff",
    "market_prob_home",
    "recent_form_diff",
    "recent_form_last5_diff",
    "recent_form_last10_diff",
    "momentum_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "back_to_back_home",
    "back_to_back_away",
    "three_in_four_home",
    "three_in_four_away",
    "market_implied_probability",
    "spread_value_signal",
    "line_movement",
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

_NBA_FEATURE_HEALTH_PRINTED = False

NBA_FEATURE_HEALTH_GROUPS: dict[str, dict[str, list[str]]] = {
    "ratings": {
        "inputs": ["offensive_rating_home", "offensive_rating_away", "defensive_rating_home", "defensive_rating_away", "net_rating_home", "net_rating_away"],
        "features": ["net_rating_diff", "pace_adjusted_scoring_diff"],
    },
    "pace": {
        "inputs": ["pace_home", "pace_away"],
        "features": ["pace_diff"],
    },
    "rest_travel": {
        "inputs": ["rest_days_home", "rest_days_away", "travel_distance_away", "timezone_shift_away", "road_trip_length_away", "travel_distance_home", "timezone_shift_home", "road_trip_length_home"],
        "features": ["rest_diff", "rest_advantage_weighted", "travel_fatigue_diff"],
    },
    "injuries": {
        "inputs": ["injury_impact_home", "injury_impact_away", "injury_impact_diff"],
        "features": ["injury_impact_diff"],
    },
    "rolling_form": {
        "inputs": ["off_rating_last5_home", "off_rating_last5_away", "def_rating_last5_home", "def_rating_last5_away", "last5_net_rating_home", "last5_net_rating_away", "last10_net_rating_home", "last10_net_rating_away"],
        "features": ["rolling_off_rating_diff_last5", "rolling_def_rating_diff_last5", "last5_net_rating_diff", "last10_net_rating_diff"],
    },
    "market": {
        "inputs": ["market_prob_home", "closing_moneyline_home", "home_odds"],
        "features": ["market_prob_home"],
    },
}


def _print_nba_feature_health(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    global _NBA_FEATURE_HEALTH_PRINTED
    if _NBA_FEATURE_HEALTH_PRINTED:
        return
    _NBA_FEATURE_HEALTH_PRINTED = True

    rows: list[dict[str, object]] = []
    for group, config in NBA_FEATURE_HEALTH_GROUPS.items():
        inputs = config["inputs"]
        features = config["features"]
        present_inputs = [col for col in inputs if col in df_before.columns]
        non_null_inputs = [col for col in present_inputs if pd.to_numeric(df_before[col], errors="coerce").notna().any()]
        feature_states = []
        for feature in features:
            series = pd.to_numeric(df_after.get(feature, pd.Series(np.nan, index=df_after.index)), errors="coerce")
            if series.isna().all():
                state = "all_nan"
            elif series.fillna(0.0).abs().sum() == 0:
                state = "all_zero"
            else:
                state = "has_signal"
            feature_states.append(f"{feature}:{state}")
        if not present_inputs:
            reason = "source_unavailable"
        elif not non_null_inputs:
            reason = "source_all_null"
        elif all("has_signal" not in state for state in feature_states):
            reason = "merge_or_transform_no_signal"
        else:
            reason = "ok"
        rows.append(
            {
                "group": group,
                "present_inputs": len(present_inputs),
                "non_null_inputs": len(non_null_inputs),
                "diagnosis": reason,
                "features": ", ".join(feature_states),
            }
        )
    print("[NBA FEATURE HEALTH]")
    print(pd.DataFrame(rows).to_string(index=False))

FEATURE_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "elo_diff": (-450.0, 450.0),
    "rest_diff": (-5.0, 5.0),
    "rest_advantage_weighted": (-8.0, 8.0),
    "travel_fatigue_diff": (-6000.0, 6000.0),
    "injury_impact_diff": (-25.0, 25.0),
    "net_rating_diff": (-35.0, 35.0),
    "pace_diff": (-15.0, 15.0),
    "rolling_off_rating_diff_last5": (-30.0, 30.0),
    "rolling_def_rating_diff_last5": (-30.0, 30.0),
    "home_away_net_rating_split_diff": (-35.0, 35.0),
    "pace_adjusted_scoring_diff": (-45.0, 45.0),
    "last5_net_rating_diff": (-40.0, 40.0),
    "last10_net_rating_diff": (-30.0, 30.0),
    "market_prob_home": (0.02, 0.98),
    "recent_form_diff": (-1.0, 1.0),
    "recent_form_last5_diff": (-1.0, 1.0),
    "recent_form_last10_diff": (-1.0, 1.0),
    "momentum_diff": (-35.0, 35.0),
    "offensive_rating_diff": (-35.0, 35.0),
    "defensive_rating_diff": (-35.0, 35.0),
    "back_to_back_home": (0.0, 1.0),
    "back_to_back_away": (0.0, 1.0),
    "three_in_four_home": (0.0, 1.0),
    "three_in_four_away": (0.0, 1.0),
    "market_implied_probability": (0.02, 0.98),
    "spread_value_signal": (-30.0, 30.0),
    "line_movement": (-25.0, 25.0),
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


def normalize_team_name(name: object) -> str:
    return str(shared_normalize_team_name(name))


def _merge_nba_team_stats(df: pd.DataFrame, nba_team_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stats = nba_team_stats.copy()
    if "team" not in stats.columns:
        raise RuntimeError("[DATA ERROR] nba_team_stats is missing required column: team")

    if "home_team_norm" not in out.columns or "away_team_norm" not in out.columns:
        if "home_team" not in out.columns or "away_team" not in out.columns:
            raise RuntimeError("[DATA ERROR] Missing team columns required for NBA stat merge")
        out["home_team_norm"] = out["home_team"].apply(normalize_team_name)
        out["away_team_norm"] = out["away_team"].apply(normalize_team_name)

    stats["team_norm"] = stats["team"].apply(normalize_team_name)
    cols_to_drop = [col for col in out.columns if "_home" in col or "_away" in col]
    out = out.drop(columns=cols_to_drop, errors="ignore")

    home_matches = out["home_team_norm"].isin(stats["team_norm"]).sum()
    away_matches = out["away_team_norm"].isin(stats["team_norm"]).sum()
    print(f"[MERGE DEBUG] Home matches: {home_matches}/{len(out)}")
    print(f"[MERGE DEBUG] Away matches: {away_matches}/{len(out)}")

    out = out.merge(
        stats.add_suffix("_home"),
        left_on="home_team_norm",
        right_on="team_norm_home",
        how="left",
    )
    out = out.merge(
        stats.add_suffix("_away"),
        left_on="away_team_norm",
        right_on="team_norm_away",
        how="left",
    )
    out["offensive_rating_home"] = out["offensive_rating_home"]
    out["offensive_rating_away"] = out["offensive_rating_away"]
    out["defensive_rating_home"] = out["defensive_rating_home"]
    out["defensive_rating_away"] = out["defensive_rating_away"]
    print("[POST MERGE SAMPLE]")
    print(out[["home_team", "offensive_rating_home", "offensive_rating_away"]].head())
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

    for col in ["offensive_rating_home", "offensive_rating_away", "defensive_rating_home", "defensive_rating_away"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(0, 110)

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
    out["offensive_rating_diff"] = _coalesce_numeric(out, ["offensive_rating_home"], default=np.nan) - _coalesce_numeric(out, ["offensive_rating_away"], default=np.nan)
    if out["offensive_rating_diff"].isna().all():
        print("[WARNING] No stat matches — fallback triggered")
        out["offensive_rating_diff"] = _coalesce_numeric(out, ["elo_home"], default=0.0) - _coalesce_numeric(out, ["elo_away"], default=0.0)
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
    feature_health_input = out.copy()

    out["elo_diff"] = _coalesce_numeric(out, ["elo_diff", "elo_home", "elo_away"], default=0.0)
    if "elo_home" in out.columns and "elo_away" in out.columns:
        out["elo_diff"] = pd.to_numeric(out["elo_home"], errors="coerce") - pd.to_numeric(out["elo_away"], errors="coerce")

    out["rest_diff"] = _coalesce_numeric(out, ["rest_diff"], default=np.nan)
    if out["rest_diff"].isna().all():
        out["rest_diff"] = _coalesce_numeric(out, ["rest_days_home"], 0.0) - _coalesce_numeric(out, ["rest_days_away"], 0.0)

    out["travel_fatigue_diff"] = _coalesce_numeric(out, ["travel_fatigue_diff"], default=np.nan)
    out["travel_fatigue_diff"] = out["travel_fatigue_diff"].astype(float)
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
    out["travel_fatigue_diff"] = out["travel_fatigue_diff"].astype(float)

    out["injury_impact_diff"] = _coalesce_numeric(out, ["injury_impact_diff"], default=np.nan)
    if out["injury_impact_diff"].isna().all():
        out["injury_impact_diff"] = _coalesce_numeric(out, ["injury_impact_home"], 0.0) - _coalesce_numeric(out, ["injury_impact_away"], 0.0)

    out["net_rating_diff"] = _coalesce_numeric(out, ["net_rating_diff"], default=np.nan)
    if out["net_rating_diff"].isna().all():
        out["net_rating_diff"] = _coalesce_numeric(out, ["net_rating_home"], 0.0) - _coalesce_numeric(out, ["net_rating_away"], 0.0)

    out["pace_diff"] = _coalesce_numeric(out, ["pace_diff"], default=np.nan)
    if out["pace_diff"].isna().all():
        out["pace_diff"] = _coalesce_numeric(out, ["pace_home"], 0.0) - _coalesce_numeric(out, ["pace_away"], 0.0)

    out["rest_advantage_weighted"] = _coalesce_numeric(out, ["rest_advantage_weighted"], default=np.nan)
    if out["rest_advantage_weighted"].isna().all():
        out["rest_advantage_weighted"] = out["rest_diff"] * (1.0 + (out["travel_fatigue_diff"].abs() / 4000.0))

    out["rolling_off_rating_diff_last5"] = _coalesce_numeric(out, ["rolling_off_rating_diff_last5"], default=np.nan)
    if out["rolling_off_rating_diff_last5"].isna().all():
        out["rolling_off_rating_diff_last5"] = (
            _coalesce_numeric(out, ["off_rating_last5_home", "rolling_off_rating_last5_home", "offensive_rating_last5_home"], 0.0)
            - _coalesce_numeric(out, ["off_rating_last5_away", "rolling_off_rating_last5_away", "offensive_rating_last5_away"], 0.0)
        )

    out["rolling_def_rating_diff_last5"] = _coalesce_numeric(out, ["rolling_def_rating_diff_last5"], default=np.nan)
    if out["rolling_def_rating_diff_last5"].isna().all():
        out["rolling_def_rating_diff_last5"] = (
            _coalesce_numeric(out, ["def_rating_last5_home", "rolling_def_rating_last5_home", "defensive_rating_last5_home"], 0.0)
            - _coalesce_numeric(out, ["def_rating_last5_away", "rolling_def_rating_last5_away", "defensive_rating_last5_away"], 0.0)
        )

    out["home_away_net_rating_split_diff"] = _coalesce_numeric(out, ["home_away_net_rating_split_diff"], default=np.nan)
    if out["home_away_net_rating_split_diff"].isna().all():
        out["home_away_net_rating_split_diff"] = (
            _coalesce_numeric(out, ["net_rating_home_split", "home_net_rating_split", "net_rating_home"], 0.0)
            - _coalesce_numeric(out, ["net_rating_away_split", "away_net_rating_split", "net_rating_away"], 0.0)
        )

    out["pace_adjusted_scoring_diff"] = _coalesce_numeric(out, ["pace_adjusted_scoring_diff"], default=np.nan)
    if out["pace_adjusted_scoring_diff"].isna().all():
        home_off = _coalesce_numeric(out, ["offensive_rating_home", "off_rating_home"], 0.0)
        away_off = _coalesce_numeric(out, ["offensive_rating_away", "off_rating_away"], 0.0)
        home_pace = _coalesce_numeric(out, ["pace_home"], 100.0).replace(0, 100.0)
        away_pace = _coalesce_numeric(out, ["pace_away"], 100.0).replace(0, 100.0)
        out["pace_adjusted_scoring_diff"] = (home_off * (home_pace / 100.0)) - (away_off * (away_pace / 100.0))

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

    out["market_implied_probability"] = _coalesce_numeric(out, ["market_implied_probability", "market_prob_home"], default=np.nan)
    if out["market_implied_probability"].isna().all():
        out["market_implied_probability"] = out["market_prob_home"]

    out["offensive_rating_diff"] = _coalesce_numeric(out, ["offensive_rating_diff"], default=np.nan)
    if out["offensive_rating_diff"].isna().all():
        out["offensive_rating_diff"] = _coalesce_numeric(out, ["offensive_rating_home", "off_rating_home"], 0.0) - _coalesce_numeric(out, ["offensive_rating_away", "off_rating_away"], 0.0)

    out["defensive_rating_diff"] = _coalesce_numeric(out, ["defensive_rating_diff"], default=np.nan)
    if out["defensive_rating_diff"].isna().all():
        out["defensive_rating_diff"] = _coalesce_numeric(out, ["defensive_rating_home", "def_rating_home"], 0.0) - _coalesce_numeric(out, ["defensive_rating_away", "def_rating_away"], 0.0)

    out["recent_form_diff"] = _coalesce_numeric(out, ["recent_form_diff"], default=np.nan)
    if out["recent_form_diff"].isna().all():
        out["recent_form_diff"] = _coalesce_numeric(out, ["recent_form_home", "win_pct_home"], 0.0) - _coalesce_numeric(out, ["recent_form_away", "win_pct_away"], 0.0)
    out["recent_form_last5_diff"] = _coalesce_numeric(out, ["recent_form_last5_diff", "last5_win_pct_diff"], default=np.nan)
    if out["recent_form_last5_diff"].isna().all():
        out["recent_form_last5_diff"] = _coalesce_numeric(out, ["last5_win_pct_home", "recent_form_last5_home"], 0.0) - _coalesce_numeric(out, ["last5_win_pct_away", "recent_form_last5_away"], 0.0)
    out["recent_form_last10_diff"] = _coalesce_numeric(out, ["recent_form_last10_diff", "last10_win_pct_diff"], default=np.nan)
    if out["recent_form_last10_diff"].isna().all():
        out["recent_form_last10_diff"] = _coalesce_numeric(out, ["last10_win_pct_home", "recent_form_last10_home"], 0.0) - _coalesce_numeric(out, ["last10_win_pct_away", "recent_form_last10_away"], 0.0)
    out["momentum_diff"] = _coalesce_numeric(out, ["momentum_diff"], default=np.nan)
    if out["momentum_diff"].isna().all():
        out["momentum_diff"] = out["last5_net_rating_diff"] - (0.5 * out["last10_net_rating_diff"])

    out["back_to_back_home"] = _coalesce_numeric(out, ["back_to_back_home"], default=np.nan)
    if out["back_to_back_home"].isna().all():
        out["back_to_back_home"] = (_coalesce_numeric(out, ["rest_days_home"], 99.0) <= 0).astype(float)
    out["back_to_back_away"] = _coalesce_numeric(out, ["back_to_back_away"], default=np.nan)
    if out["back_to_back_away"].isna().all():
        out["back_to_back_away"] = (_coalesce_numeric(out, ["rest_days_away"], 99.0) <= 0).astype(float)
    out["three_in_four_home"] = _coalesce_numeric(out, ["three_in_four_home"], default=np.nan)
    if out["three_in_four_home"].isna().all():
        out["three_in_four_home"] = (_coalesce_numeric(out, ["games_last_four_days_home"], 0.0) >= 3).astype(float)
    out["three_in_four_away"] = _coalesce_numeric(out, ["three_in_four_away"], default=np.nan)
    if out["three_in_four_away"].isna().all():
        out["three_in_four_away"] = (_coalesce_numeric(out, ["games_last_four_days_away"], 0.0) >= 3).astype(float)
    out["spread_value_signal"] = _coalesce_numeric(out, ["spread_value_signal"], default=np.nan)
    if out["spread_value_signal"].isna().all():
        out["spread_value_signal"] = _coalesce_numeric(out, ["model_spread", "projected_spread"], 0.0) - _coalesce_numeric(out, ["spread", "spread_line"], 0.0)
    out["line_movement"] = _coalesce_numeric(out, ["line_movement"], default=np.nan)
    if out["line_movement"].isna().all():
        out["line_movement"] = _coalesce_numeric(out, ["closing_spread_home", "spread_line"], 0.0) - _coalesce_numeric(out, ["opening_spread_home", "spread_line"], 0.0)

    for column in NBA_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if column == "market_prob_home":
            out[column] = out[column].fillna(0.5)
        else:
            out[column] = out[column].fillna(0.0)

        lo, hi = FEATURE_CLIP_BOUNDS[column]
        out[column] = out[column].clip(lo, hi)

    _print_nba_feature_health(feature_health_input, out)
    _print_nba_neutral_feature_reasons(feature_health_input, out)
    return out


def _print_nba_neutral_feature_reasons(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    checked = [
        "recent_form_diff",
        "recent_form_last5_diff",
        "recent_form_last10_diff",
        "momentum_diff",
        "offensive_rating_diff",
        "defensive_rating_diff",
        "net_rating_diff",
        "rest_diff",
        "back_to_back_home",
        "back_to_back_away",
        "three_in_four_home",
        "three_in_four_away",
        "travel_fatigue_diff",
        "market_implied_probability",
        "spread_value_signal",
        "line_movement",
    ]
    rows = []
    for feature in checked:
        series = pd.to_numeric(df_after.get(feature, pd.Series(np.nan, index=df_after.index)), errors="coerce")
        state = "real" if series.notna().any() and series.fillna(0.0).abs().sum() != 0 else "neutral"
        reason = "has non-zero calculated signal" if state == "real" else "source columns unavailable or calculate to neutral zero"
        rows.append({"feature": feature, "status": state, "reason": reason})
    print("[NBA FEATURE QUALITY DETAIL]")
    print(pd.DataFrame(rows).to_string(index=False))
