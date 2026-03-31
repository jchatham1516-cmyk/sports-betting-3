"""Data loading helpers for live Odds API ingestion with explicit test-mode fallbacks."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sports_betting.sports.common.team_names import normalize_team_name
from sports_betting.sports.common.feature_engineering import SPORT_EFFICIENCY_FEATURES, add_elo_features, enrich_with_context_features
from sports_betting.sports.common.game_filters import current_sports_day_window, filter_games_window
from sports_betting.sports.mlb.build_historical import build_mlb_historical
from sports_betting.sports.mlb.schema import MLB_HISTORICAL_COLUMNS, MLB_REQUIRED_FEATURES

from sports_betting.scripts.build_nba_historical_dataset import NBA_HISTORICAL_COLUMNS


ROOT = Path(__file__).resolve().parents[1] / "data"
LOGGER = logging.getLogger(__name__)
SUPPORTED_SPORTS = ("nba", "nfl", "nhl", "mlb", "soccer")
OPTIONAL_SPORTS = {"mlb", "soccer"}
ALLOW_MINIMAL_DATASET = True
REQUIRED_DEFAULT_COLUMNS = [
    "home_cover",
    "over_hit",
    "spread_line",
    "total_line",
]

SPORT_TO_ODDS_API_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
    "soccer": "soccer_usa_mls",
}

# Shared training feature space used by all sports models.
BASE_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "travel_distance",
    "travel_fatigue_diff",
    "injury_impact",
    "injury_impact_diff",
    "injury_impact_home",
    "injury_impact_away",
    "starter_out_count_home",
    "starter_out_count_away",
    "star_player_out_flag_home",
    "star_player_out_flag_away",
    "starting_goalie_out_flag_home",
    "starting_goalie_out_flag_away",
    "qb_out_flag_home",
    "qb_out_flag_away",
    "offensive_injury_weight_diff",
    "defensive_injury_weight_diff",
    "offensive_injury_weight_home",
    "offensive_injury_weight_away",
    "defensive_injury_weight_home",
    "defensive_injury_weight_away",
    "injury_confidence_score",
    "injury_data_stale_flag",
    "market_prob",
    "model_prob",
    "adjusted_prob",
    "edge",
    "expected_value",
    "line_movement",
    "clv_placeholder",
    "opening_line",
    "bet_line",
    "current_line",
    "closing_line",
    "clv_diff",
    "public_favorite_bias_flag",
    "favorite_inflation_flag",
    "underdog_inflation_flag",
    "rest_days_home",
    "rest_days_away",
    "back_to_back_home",
    "back_to_back_away",
    "three_games_in_four_home",
    "three_games_in_four_away",
    "road_trip_length_home",
    "road_trip_length_away",
    "timezone_shift_home",
    "timezone_shift_away",
    "travel_distance_home",
    "travel_distance_away",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "net_rating_diff",
    "pace",
    "home_indicator",
    "rest_days",
    "back_to_back",
    "three_games_in_four",
    "time_zone_shift",
    "starter_out_count",
    "star_player_out_flag",
    "qb_out_flag",
    "starting_goalie_out_flag",
    "public_bias_flag",
    "last5_net_rating_diff",
    "last10_net_rating_diff",
    "recent_goal_diff",
    "recent_epa_diff",
    "pitcher_era_home",
    "pitcher_era_away",
    "pitcher_diff",
    "goalie_save_home",
    "goalie_save_away",
    "goalie_diff",
    "adjusted_edge",
]

FEATURE_COLUMNS_BY_SPORT = {
    "nba": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nba"]],
    "nfl": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nfl"]],
    "nhl": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nhl"]],
    "mlb": list(BASE_FEATURE_COLUMNS) + list(MLB_REQUIRED_FEATURES),
    "soccer": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES.get("nfl", [])],
}
TARGET_COLUMNS = ["home_win", "home_cover", "over_hit"]
NBA_CORE_REQUIRED_COLUMNS = [
    "home_win",
    "home_cover",
    "over_hit",
    "closing_moneyline_home",
    "closing_spread_home",
    "closing_total",
    "elo_diff",
    "rest_diff",
    "injury_impact_diff",
    "net_rating_diff",
    "pace_diff",
    "top_rotation_eff_diff",
]


def historical_file_path(sport: str) -> Path:
    return ROOT / "historical" / f"{sport}_historical.csv"


def model_artifact_path(sport: str) -> Path:
    return ROOT / "models" / f"{sport}_model.pkl"


def required_historical_columns(sport: str) -> list[str]:
    if sport not in FEATURE_COLUMNS_BY_SPORT:
        raise ValueError(f"Unsupported sport '{sport}'. Supported sports: {', '.join(sorted(FEATURE_COLUMNS_BY_SPORT))}")
    if sport == "nba":
        return sorted(set(NBA_HISTORICAL_COLUMNS + FEATURE_COLUMNS_BY_SPORT[sport] + ["spread_line", "total_line"]))
    if sport == "mlb":
        return sorted(set(MLB_HISTORICAL_COLUMNS))
    core = ["date", "home_team", "away_team", "home_score", "away_score", "closing_moneyline_home", "closing_moneyline_away", "closing_spread_home", "closing_total"]
    return sorted(set(core + FEATURE_COLUMNS_BY_SPORT[sport] + ["spread_line", "total_line"] + TARGET_COLUMNS))


def _normalize_sports(sports: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = list(sports) if sports else list(SUPPORTED_SPORTS)
    unknown = [sport for sport in normalized if sport not in SUPPORTED_SPORTS]
    if unknown:
        raise ValueError(f"Unsupported sport(s): {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_SPORTS)}")
    return normalized


def _coalesce_numeric(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in REQUIRED_DEFAULT_COLUMNS if col not in df.columns]
    if missing_cols:
        filler_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, filler_df], axis=1)
        print(f"[DATA FIX] Added missing columns: {missing_cols}")
    return df


def _american_to_prob(odds: pd.Series | float | int) -> pd.Series:
    series = pd.to_numeric(odds, errors="coerce")
    return series.apply(
        lambda x: abs(x) / (abs(x) + 100) if x < 0 else (100 / (x + 100) if x > 0 else 0.5)
    )


def _standardize_historical_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    out = df.copy()
    rename_block: dict[str, pd.Series] = {}
    for team_col in ("team", "home_team", "away_team"):
        if team_col in out.columns:
            rename_block[team_col] = out[team_col].apply(_normalize_team_name)
    if rename_block:
        out = out.assign(**rename_block)

    home_moneyline = (
        _coalesce_numeric(out, ["home_moneyline", "closing_moneyline_home", "home_odds"])
        if {"home_moneyline", "closing_moneyline_home", "home_odds"} & set(out.columns)
        else pd.Series(0.0, index=out.index, dtype="float64")
    )
    spread = (
        _coalesce_numeric(out, ["spread", "closing_spread_home", "spread_line"])
        if {"spread", "closing_spread_home", "spread_line"} & set(out.columns)
        else pd.Series(0.0, index=out.index, dtype="float64")
    )

    new_features = pd.DataFrame(
        {
            "home_moneyline": home_moneyline,
            "implied_home_prob": _american_to_prob(home_moneyline),
            "spread": spread,
            "spread_abs": pd.to_numeric(spread, errors="coerce").fillna(0.0).abs(),
            "is_favorite": (pd.to_numeric(home_moneyline, errors="coerce").fillna(0.0) < 0).astype(int),
            "elo_diff": _coalesce_numeric(out, ["elo_diff"]),
            "rest_diff": _coalesce_numeric(out, ["rest_diff"]),
            "travel_distance": _coalesce_numeric(out, ["travel_distance", "travel_fatigue_diff", "travel_diff"]),
            "offensive_rating_diff": _coalesce_numeric(out, ["offensive_rating_diff", "off_rating_diff", "epa_per_play_diff", "xgf_diff", "off_rating_diff"]),
            "defensive_rating_diff": _coalesce_numeric(out, ["defensive_rating_diff", "def_rating_diff", "xga_diff", "pressure_rate_diff"]),
            "net_rating_diff": _coalesce_numeric(out, ["net_rating_diff", "success_rate_diff", "special_teams_diff"]),
            "injury_impact": _coalesce_numeric(out, ["injury_impact", "injury_impact_diff", "qb_impact_diff", "goalie_strength_diff"]),
            "pace": _coalesce_numeric(out, ["pace", "pace_diff", "xg_total", "weather_total_impact"]),
            "home_indicator": _coalesce_numeric(out, ["home_indicator"], default=1.0).clip(0, 1),
            "travel_fatigue_diff": _coalesce_numeric(out, ["travel_fatigue_diff"]),
            "injury_impact_home": _coalesce_numeric(out, ["injury_impact_home"]),
            "injury_impact_away": _coalesce_numeric(out, ["injury_impact_away"]),
            "injury_impact_diff": _coalesce_numeric(out, ["injury_impact_diff", "injury_impact"]),
            "starter_out_count_home": _coalesce_numeric(out, ["starter_out_count_home"]),
            "starter_out_count_away": _coalesce_numeric(out, ["starter_out_count_away"]),
            "star_player_out_flag_home": _coalesce_numeric(out, ["star_player_out_flag_home"]),
            "star_player_out_flag_away": _coalesce_numeric(out, ["star_player_out_flag_away"]),
            "starting_goalie_out_flag_home": _coalesce_numeric(out, ["starting_goalie_out_flag_home"]),
            "starting_goalie_out_flag_away": _coalesce_numeric(out, ["starting_goalie_out_flag_away"]),
            "qb_out_flag_home": _coalesce_numeric(out, ["qb_out_flag_home"]),
            "qb_out_flag_away": _coalesce_numeric(out, ["qb_out_flag_away"]),
            "offensive_injury_weight_diff": _coalesce_numeric(out, ["offensive_injury_weight_diff"]),
            "defensive_injury_weight_diff": _coalesce_numeric(out, ["defensive_injury_weight_diff"]),
            "offensive_injury_weight_home": _coalesce_numeric(out, ["offensive_injury_weight_home"]),
            "offensive_injury_weight_away": _coalesce_numeric(out, ["offensive_injury_weight_away"]),
            "defensive_injury_weight_home": _coalesce_numeric(out, ["defensive_injury_weight_home"]),
            "defensive_injury_weight_away": _coalesce_numeric(out, ["defensive_injury_weight_away"]),
            "injury_confidence_score": _coalesce_numeric(out, ["injury_confidence_score"], default=0.5),
            "injury_data_stale_flag": _coalesce_numeric(out, ["injury_data_stale_flag"]),
            "market_prob": _coalesce_numeric(out, ["market_prob"]),
            "model_prob": _coalesce_numeric(out, ["model_prob"]),
            "adjusted_prob": _coalesce_numeric(out, ["adjusted_prob"], default=np.nan),
            "pitcher_era_home": _coalesce_numeric(out, ["pitcher_era_home"]),
            "pitcher_era_away": _coalesce_numeric(out, ["pitcher_era_away"]),
            "pitcher_diff": _coalesce_numeric(out, ["pitcher_diff", "starter_rating_diff"]),
            "goalie_save_home": _coalesce_numeric(out, ["goalie_save_home", "goalie_save_strength_home"], default=np.nan).astype(float).fillna(0.905),
            "goalie_save_away": _coalesce_numeric(out, ["goalie_save_away", "goalie_save_strength_away"], default=np.nan).astype(float).fillna(0.905),
            "expected_value": _coalesce_numeric(out, ["expected_value"]),
            "line_movement": _coalesce_numeric(out, ["line_movement"]),
            "clv_placeholder": _coalesce_numeric(out, ["clv_placeholder"]),
            "public_favorite_bias_flag": _coalesce_numeric(out, ["public_favorite_bias_flag", "public_bias_flag"]),
            "public_bias_flag": _coalesce_numeric(out, ["public_bias_flag", "public_favorite_bias_flag"]),
            "favorite_inflation_flag": _coalesce_numeric(out, ["favorite_inflation_flag"]),
            "underdog_inflation_flag": _coalesce_numeric(out, ["underdog_inflation_flag"]),
            "rest_days_home": _coalesce_numeric(out, ["rest_days_home"]),
            "rest_days_away": _coalesce_numeric(out, ["rest_days_away"]),
            "back_to_back_home": _coalesce_numeric(out, ["back_to_back_home"]),
            "back_to_back_away": _coalesce_numeric(out, ["back_to_back_away"]),
            "three_games_in_four_home": _coalesce_numeric(out, ["three_games_in_four_home", "three_in_four_home"]),
            "three_games_in_four_away": _coalesce_numeric(out, ["three_games_in_four_away", "three_in_four_away"]),
            "three_games_in_four": _coalesce_numeric(out, ["three_games_in_four", "three_in_four"]),
            "road_trip_length_home": _coalesce_numeric(out, ["road_trip_length_home"]),
            "road_trip_length_away": _coalesce_numeric(out, ["road_trip_length_away"]),
            "timezone_shift_home": _coalesce_numeric(out, ["timezone_shift_home"]),
            "timezone_shift_away": _coalesce_numeric(out, ["timezone_shift_away"]),
            "time_zone_shift": _coalesce_numeric(out, ["time_zone_shift", "timezone_shift_away"]),
            "travel_distance_home": _coalesce_numeric(out, ["travel_distance_home"]),
            "travel_distance_away": _coalesce_numeric(out, ["travel_distance_away"]),
            "rest_days": _coalesce_numeric(out, ["rest_days", "rest_days_away"]),
            "back_to_back": _coalesce_numeric(out, ["back_to_back", "back_to_back_away"]),
            "starter_out_count": _coalesce_numeric(out, ["starter_out_count", "starter_out_count_away"]),
            "star_player_out_flag": _coalesce_numeric(out, ["star_player_out_flag", "star_player_out_flag_away"]),
            "qb_out_flag": _coalesce_numeric(out, ["qb_out_flag", "qb_out_flag_away"]),
            "starting_goalie_out_flag": _coalesce_numeric(out, ["starting_goalie_out_flag", "starting_goalie_out_flag_away"]),
            "last5_net_rating_diff": _coalesce_numeric(out, ["last5_net_rating_diff"]),
            "last10_net_rating_diff": _coalesce_numeric(out, ["last10_net_rating_diff"]),
            "point_diff_home": _coalesce_numeric(out, ["point_diff_home", "avg_point_diff_home", "score_diff_home"]),
            "point_diff_away": _coalesce_numeric(out, ["point_diff_away", "avg_point_diff_away", "score_diff_away"]),
            "last5_net_rating_home": _coalesce_numeric(out, ["last5_net_rating_home", "last5_home", "net_rating_home"]),
            "last5_net_rating_away": _coalesce_numeric(out, ["last5_net_rating_away", "last5_away", "net_rating_away"]),
            "last10_net_rating_home": _coalesce_numeric(out, ["last10_net_rating_home", "last10_home", "last5_net_rating_home"]),
            "last10_net_rating_away": _coalesce_numeric(out, ["last10_net_rating_away", "last10_away", "last5_net_rating_away"]),
            "recent_goal_diff": _coalesce_numeric(out, ["recent_goal_diff"]),
            "recent_epa_diff": _coalesce_numeric(out, ["recent_epa_diff"]),
            "spread_line": _coalesce_numeric(out, ["spread_line", "closing_spread_home"]),
            "total_line": _coalesce_numeric(out, ["total_line", "closing_total"]),
            "home_score": _coalesce_numeric(out, ["home_score"]),
            "away_score": _coalesce_numeric(out, ["away_score"]),
            "date": pd.to_datetime(out.get("date", out.get("event_date")), errors="coerce", utc=True),
            "event_date": pd.to_datetime(out.get("event_date", out.get("date")), errors="coerce", utc=True),
            "closing_moneyline_home": _coalesce_numeric(out, ["closing_moneyline_home", "home_odds"]),
            "closing_moneyline_away": _coalesce_numeric(out, ["closing_moneyline_away", "away_odds"]),
            "closing_spread_home": _coalesce_numeric(out, ["closing_spread_home", "spread_line"]),
            "closing_total": _coalesce_numeric(out, ["closing_total", "total_line"]),
            "opening_line": _coalesce_numeric(out, ["opening_line", "open_line", "spread_line"]),
            "bet_line": _coalesce_numeric(out, ["bet_line", "spread_line"]),
            "current_line": _coalesce_numeric(out, ["current_line", "spread_line"]),
            "closing_line": _coalesce_numeric(out, ["closing_line", "spread_line"]),
            "home_win": _coalesce_numeric(out, ["home_win"]).round().clip(0, 1).astype(int),
            "home_cover": _coalesce_numeric(out, ["home_cover"]).round().clip(0, 1).astype(int),
            "over_hit": _coalesce_numeric(out, ["over_hit"]).round().clip(0, 1).astype(int),
        },
        index=out.index,
    )
    new_features["model_prob_before_injury"] = new_features["model_prob"]
    new_features["model_prob"] = (new_features["model_prob"] + (new_features["injury_impact_diff"] * 0.03)).clip(0.05, 0.95)
    new_features["model_prob_after_injury"] = new_features["model_prob"]
    missing_adjusted = new_features["adjusted_prob"].isna()
    if missing_adjusted.any():
        new_features.loc[missing_adjusted, "adjusted_prob"] = new_features.loc[missing_adjusted, "model_prob"]
    new_features["edge"] = (new_features["model_prob"] - new_features["market_prob"]) * 1.5
    new_features["goalie_diff"] = new_features["goalie_save_home"] - new_features["goalie_save_away"]
    sport_series = out.get("sport", pd.Series("", index=out.index)).astype(str).str.lower()
    pitcher_weight = pd.Series(0.03, index=out.index, dtype="float64")
    pitcher_weight.loc[sport_series.eq("mlb")] = 0.015
    new_features["adjusted_edge"] = new_features["edge"] + (new_features["pitcher_diff"] * pitcher_weight) + (new_features["goalie_diff"] * 0.05)
    new_features["point_diff_diff"] = new_features["point_diff_home"] - new_features["point_diff_away"]
    new_features["recent_form_diff"] = new_features["last5_net_rating_home"] - new_features["last5_net_rating_away"]
    new_features["momentum_diff"] = (
        (new_features["last10_net_rating_home"] - new_features["last5_net_rating_home"])
        - (new_features["last10_net_rating_away"] - new_features["last5_net_rating_away"])
    )
    new_features["power_rating_home"] = (_coalesce_numeric(out, ["elo_home"], default=1500.0) * 0.6) + (_coalesce_numeric(out, ["net_rating_home"], default=0.0) * 0.4)
    new_features["power_rating_away"] = (_coalesce_numeric(out, ["elo_away"], default=1500.0) * 0.6) + (_coalesce_numeric(out, ["net_rating_away"], default=0.0) * 0.4)
    new_features["power_rating_diff"] = new_features["power_rating_home"] - new_features["power_rating_away"]
    new_features["clv_diff"] = new_features["closing_line"] - new_features["bet_line"]

    columns_to_replace = [c for c in new_features.columns if c in out.columns]
    if columns_to_replace:
        out = out.drop(columns=columns_to_replace)
    out = pd.concat([out, new_features], axis=1)

    numeric_cols = [col for col in out.columns if col not in {"date", "event_date", "season", "game_id", "home_team", "away_team", "sport", "team"}]
    if numeric_cols:
        numeric_frame = pd.DataFrame(
            {col: pd.to_numeric(out[col], errors="coerce").fillna(0) for col in numeric_cols},
            index=out.index,
        )
        out = out.drop(columns=numeric_cols)
        out = pd.concat([out, numeric_frame], axis=1)

    out = add_elo_features(out, sport)
    LOGGER.info("[%s] Standardized historical feature columns for training.", sport.upper())
    return out


def load_historical_dataset(sport: str) -> pd.DataFrame:
    hist_path = historical_file_path(sport)
    if not hist_path.exists():
        print(f"[{sport.upper()}] ⚠️ Missing CSV — using model artifact instead")
        return pd.DataFrame()
    df = pd.read_csv(hist_path)
    required_columns = required_historical_columns(sport)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[{sport.upper()}] ⚠️ Missing columns: {missing}")
    if missing:
        filler_df = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, filler_df], axis=1)
    df = ensure_required_columns(df)
    return _standardize_historical_features(df, sport)


def load_nba_historical_dataset() -> pd.DataFrame:
    hist_path = historical_file_path("nba")
    if not hist_path.exists():
        print("[NBA] ⚠️ Missing CSV — using model artifact instead")
        return pd.DataFrame()

    df = pd.read_csv(hist_path)
    required_columns = required_historical_columns("nba")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[NBA] ⚠️ Missing columns: {missing}")
        filler_df = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, filler_df], axis=1)
    df = ensure_required_columns(df)
    missing_core = [c for c in NBA_CORE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_core:
        print("[NBA] ⚠️ Historical validation warning — continuing with fallback models")
        filler_df = pd.DataFrame(0, index=df.index, columns=missing_core)
        df = pd.concat([df, filler_df], axis=1)

    # Keep the full NBA schema stable even when optional sources are absent.
    nba_missing = [col for col in NBA_HISTORICAL_COLUMNS if col not in df.columns]
    if nba_missing:
        filler_df = pd.DataFrame(0.0, index=df.index, columns=nba_missing)
        df = pd.concat([df, filler_df], axis=1)
    feature_missing = [col for col in FEATURE_COLUMNS_BY_SPORT["nba"] if col not in df.columns]
    if feature_missing:
        filler_df = pd.DataFrame(0.0, index=df.index, columns=feature_missing)
        df = pd.concat([df, filler_df], axis=1)

    standardized = _standardize_historical_features(df, "nba")
    nba_numeric_cols = [col for col in NBA_HISTORICAL_COLUMNS if col not in {"date", "season", "game_id", "home_team", "away_team"}]
    if nba_numeric_cols:
        standardized_numeric = pd.DataFrame(
            {
                col: pd.to_numeric(standardized.get(col, 0.0), errors="coerce").fillna(0.0)
                for col in nba_numeric_cols
            },
            index=standardized.index,
        )
        standardized = standardized.drop(columns=[c for c in nba_numeric_cols if c in standardized.columns])
        standardized = pd.concat([standardized, standardized_numeric], axis=1)
    return standardized


def load_nfl_historical_dataset() -> pd.DataFrame:
    return load_historical_dataset("nfl")


def load_nhl_historical_dataset() -> pd.DataFrame:
    return load_historical_dataset("nhl")


def validate_historical_requirements(
    sports: list[str] | tuple[str, ...] | None = None,
    allow_model_artifacts: bool = True,
    validate_schema: bool = True,
) -> None:
    sports_to_check = _normalize_sports(sports)
    missing_errors: list[str] = []
    warnings: list[str] = []
    schema_messages: list[str] = []

    for sport in sports_to_check:
        hist_path = historical_file_path(sport)
        artifact_path = model_artifact_path(sport)
        hist_exists = hist_path.exists()
        artifact_exists = artifact_path.exists()

        has_csv = hist_exists
        has_model = allow_model_artifacts and artifact_exists

        if not has_csv and not has_model:
            if sport.lower() in OPTIONAL_SPORTS:
                warnings.append(f"[{sport.upper()}] Skipping (no data/model yet)")
            else:
                missing_errors.append(
                    f"- {sport.upper()}: missing both historical CSV ({hist_path}) and trained model artifact ({artifact_path})."
                )
            continue

        if validate_schema and hist_exists:
            raw_df = pd.read_csv(hist_path, nrows=5)
            required_cols = required_historical_columns(sport)
            missing_cols = [col for col in required_cols if col not in raw_df.columns]
            missing_columns = sorted(set(missing_cols))
            if missing_columns:
                print(f"[{sport.upper()}] ⚠️ Missing columns: {missing_columns}")

    if warnings:
        for warning in warnings:
            print(warning)

    if missing_errors or schema_messages:
        details = [
            "Historical data preflight failed.",
            "Required historical CSV files:",
            *(f"- {sport.upper()}: {historical_file_path(sport)}" for sport in sports_to_check),
            "",
            "Required trained model artifacts (alternative to CSV at runtime):",
            *(f"- {sport.upper()}: {model_artifact_path(sport)}" for sport in sports_to_check),
            "",
            "Required historical CSV schema by sport:",
        ]
        for sport in sports_to_check:
            details.append(f"- {sport.upper()}: {', '.join(required_historical_columns(sport))}")
        if missing_errors:
            details.extend(["", "Missing data/model files:", *missing_errors])
        if schema_messages:
            details.extend(["", "Schema issues:", *schema_messages])
        print("⚠️ Historical validation warning — continuing with fallback models")
        print("\n".join(details))
        return


def validate_model_artifacts_exist(
    sports: list[str] | tuple[str, ...] | None = None,
) -> None:
    sports_to_check = _normalize_sports(sports)
    missing = [sport for sport in sports_to_check if not model_artifact_path(sport).exists()]
    if not missing:
        return

    details = [
        "Model artifact preflight failed.",
        "Required trained model artifacts:",
        *(f"- {sport.upper()}: {model_artifact_path(sport)}" for sport in sports_to_check),
        "",
        "Missing artifacts:",
        *(f"- {sport.upper()}: {model_artifact_path(sport)}" for sport in missing),
        "",
        "Generate missing artifacts with:",
        f"- python -m sports_betting.scripts.train_models --sports {' '.join(missing)}",
    ]
    raise RuntimeError("\n".join(details))


def _is_test_mode() -> bool:
    return os.getenv("TEST_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def is_test_mode() -> bool:
    return _is_test_mode()


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def generate_sample_data(sport: str, rows: int = 300) -> pd.DataFrame:
    """Generate synthetic feature-rich data for explicit test mode only."""
    rng = np.random.default_rng(42)
    base = pd.DataFrame(
        {
            "game_id": [f"{sport}_{i}" for i in range(rows)],
            "away_team": [f"A{i%30}" for i in range(rows)],
            "home_team": [f"H{i%30}" for i in range(rows)],
            "event_date": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "date": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "spread_line": rng.normal(-2, 4, rows),
            "total_line": rng.normal(220 if sport == "nba" else 44 if sport == "nfl" else 6.0, 8, rows),
            "away_odds": rng.choice([-130, -120, -110, 100, 115, 130], rows),
            "home_odds": rng.choice([-130, -120, -110, 100, 115, 130], rows),
            "away_spread_odds": rng.choice([-112, -110, -108, 100], rows),
            "home_spread_odds": rng.choice([-112, -110, -108, 100], rows),
            "over_odds": rng.choice([-112, -110, -108, 100], rows),
            "under_odds": rng.choice([-112, -110, -108, 100], rows),
            "sportsbook": ["synthetic"] * rows,
            "home_indicator": np.ones(rows),
            "home_score": rng.integers(70,130,size=rows),
            "away_score": rng.integers(70,130,size=rows),
        }
    )
    for c in BASE_FEATURE_COLUMNS:
        if c == "home_indicator":
            continue
        base[c] = rng.normal(0, 1, rows)
    for c in FEATURE_COLUMNS_BY_SPORT.get(sport, []):
        if c not in base.columns and c != "home_indicator":
            base[c] = rng.normal(0, 1, rows)

    win_score = 0.25 * base["elo_diff"] + 0.2 * base["net_rating_diff"] + 0.1 * base["rest_diff"] - 0.08 * base["travel_distance"]
    spread_score = 0.18 * base["net_rating_diff"] + 0.08 * base["rest_diff"] - 0.15 * base["spread_line"]
    total_score = 0.22 * base["pace"] + 0.16 * base["offensive_rating_diff"] - 0.12 * base["defensive_rating_diff"]
    base["home_win"] = (win_score + rng.normal(0, 0.8, rows) > 0).astype(int)
    base["closing_moneyline_home"] = base["home_odds"]
    base["closing_moneyline_away"] = base["away_odds"]
    base["closing_spread_home"] = base["spread_line"]
    base["closing_total"] = base["total_line"]
    base["home_cover"] = (spread_score + rng.normal(0, 0.8, rows) > 0).astype(int)
    base["over_hit"] = (total_score + rng.normal(0, 0.8, rows) > 0).astype(int)
    return base


def clean_team_name(name: str | None) -> str:
    return (
        str(name or "").lower()
        .strip()
        .replace(".", "")
        .replace("-", " ")
    )


def _normalize_team_name(name: str | None) -> str:
    return normalize_team_name(clean_team_name(str(name or "")))


def _build_outcome_lookup(outcomes: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for out in outcomes:
        normalized = _normalize_team_name(out.get("name"))
        if normalized:
            lookup[normalized] = out
    return lookup


def _extract_market_prices(event: dict, market_key: str) -> dict[str, int | float | str] | None:
    raw_home = event.get("home_team")
    raw_away = event.get("away_team")
    parsed_home = _normalize_team_name(raw_home)
    parsed_away = _normalize_team_name(raw_away)

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != market_key:
                continue
            outcomes_list = market.get("outcomes", [])
            outcomes = {out.get("name"): out for out in outcomes_list}
            normalized_outcomes = _build_outcome_lookup(outcomes_list)

            if market_key == "h2h":
                home_outcome = outcomes.get(raw_home) or normalized_outcomes.get(parsed_home)
                away_outcome = outcomes.get(raw_away) or normalized_outcomes.get(parsed_away)
                if not (home_outcome and away_outcome):
                    continue
                draw_outcome = outcomes.get("Draw") or outcomes.get("draw")
                return {
                    "sportsbook": book.get("key", "unknown"),
                    "home_odds": int(home_outcome["price"]),
                    "away_odds": int(away_outcome["price"]),
                    "draw_odds": int(draw_outcome["price"]) if draw_outcome else 0,
                    "sportsbook_event_home_team": str(raw_home or ""),
                    "sportsbook_event_away_team": str(raw_away or ""),
                    "sportsbook_home_outcome_name": str(home_outcome.get("name", "")),
                    "sportsbook_away_outcome_name": str(away_outcome.get("name", "")),
                }

            if market_key == "spreads":
                home = outcomes.get(raw_home) or normalized_outcomes.get(parsed_home)
                away = outcomes.get(raw_away) or normalized_outcomes.get(parsed_away)
                if not (home and away):
                    continue
                return {
                    "sportsbook": book.get("key", "unknown"),
                    "home_spread_odds": int(home["price"]),
                    "away_spread_odds": int(away["price"]),
                    "spread_line": float(home.get("point", 0.0)),
                }
            if market_key == "totals":
                over = outcomes.get("Over")
                under = outcomes.get("Under")
                if over and under:
                    return {
                        "sportsbook": book.get("key", "unknown"),
                        "over_odds": int(over["price"]),
                        "under_odds": int(under["price"]),
                        "total_line": float(over.get("point", 0.0)),
                    }
    return None


def fetch_live_daily_odds(sport: str, today_only: bool = True) -> pd.DataFrame:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required in live mode. Set TEST_MODE=true to use synthetic demo data.")

    odds_sport = SPORT_TO_ODDS_API_KEY.get(sport)
    if not odds_sport:
        raise ValueError(f"Unsupported sport '{sport}'. Supported sports: {', '.join(sorted(SPORT_TO_ODDS_API_KEY))}")

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if today_only:
        start, end = current_sports_day_window()
        params["commenceTimeFrom"] = format_odds_api_time(start)
        params["commenceTimeTo"] = format_odds_api_time(end)

    query = urlencode(params)
    url = f"https://api.the-odds-api.com/v4/sports/{odds_sport}/odds/?{query}"

    LOGGER.info("[%s] Odds API enabled: True (provider=the-odds-api, sport_key=%s)", sport.upper(), odds_sport)
    try:
        with urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Odds API request failed for {sport}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Odds API request failed for {sport}: {exc.reason}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Odds API payload for {sport}: expected list, got {type(payload).__name__}")

    events = filter_games_window(payload) if today_only else payload

    records: list[dict] = []
    for event in events:
        h2h = _extract_market_prices(event, "h2h")
        spreads = _extract_market_prices(event, "spreads")
        totals = _extract_market_prices(event, "totals")
        if not (h2h and spreads and totals):
            continue

        record = {
            "game_id": event.get("id") or f"{sport}_{len(records)}",
            "away_team": event.get("away_team", "UNKNOWN_AWAY"),
            "home_team": event.get("home_team", "UNKNOWN_HOME"),
            "event_date": event.get("commence_time"),
            "commence_time": event.get("commence_time"),
            "sportsbook": h2h.get("sportsbook") or spreads.get("sportsbook") or totals.get("sportsbook") or "unknown",
        }
        record.update(h2h)
        record.update(spreads)
        record.update(totals)
        record["home_indicator"] = 1.0

        LOGGER.info(
            "[%s] odds map | raw_away=%s raw_home=%s parsed_away=%s parsed_home=%s sportsbook_event_away=%s sportsbook_event_home=%s away_ml=%s home_ml=%s away_spread=%s@%s home_spread=%s@%s",
            sport.upper(),
            event.get("away_team"),
            event.get("home_team"),
            _normalize_team_name(event.get("away_team")),
            _normalize_team_name(event.get("home_team")),
            h2h.get("sportsbook_event_away_team"),
            h2h.get("sportsbook_event_home_team"),
            record.get("away_odds"),
            record.get("home_odds"),
            record.get("away_spread_odds"),
            -float(record.get("spread_line", 0.0)),
            record.get("home_spread_odds"),
            float(record.get("spread_line", 0.0)),
        )

        for feature_name in FEATURE_COLUMNS_BY_SPORT[sport]:
            record.setdefault(feature_name, 0.0)
        records.append(record)

    daily = pd.DataFrame(records)
    if daily.empty:
        print(f"[{sport.upper()}] No games available in the next 48 hours — skipping.")
        return pd.DataFrame()

    daily["event_date"] = pd.to_datetime(daily["event_date"], utc=True, errors="coerce")
    daily["commence_time"] = pd.to_datetime(daily["commence_time"], utc=True, errors="coerce")
    if daily["event_date"].isna().any():
        bad_count = int(daily["event_date"].isna().sum())
        raise RuntimeError(f"[{sport.upper()}] Failed to parse event_date for {bad_count} events from Odds API.")

    LOGGER.info(
        "[%s] Odds API events returned: %s (events after date filter: %s, usable with full markets: %s)",
        sport.upper(),
        len(payload),
        len(events),
        len(daily),
    )
    return daily


def format_odds_api_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_historical_and_daily(sport: str, today_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_path = historical_file_path(sport)
    daily_path = ROOT / "raw" / f"{sport}_daily.csv"
    print(f"[{sport.upper()}] Looking for historical CSV at: {hist_path}")
    print(f"[{sport.upper()}] Exists: {hist_path.exists()}")

    historical = load_csv_or_empty(hist_path)
    if not historical.empty:
        required_columns = required_historical_columns(sport)
        missing = [col for col in required_columns if col not in historical.columns]
        if missing:
            print(f"[{sport.upper()}] ⚠️ Missing columns: {missing}")
        missing_cols = [col for col in required_columns if col not in historical.columns]
        if missing_cols:
            filler = pd.DataFrame(0, index=historical.index, columns=missing_cols)
            historical = pd.concat([historical, filler], axis=1)
        historical = ensure_required_columns(historical)
        historical = _standardize_historical_features(historical, sport)

    if _is_test_mode():
        LOGGER.warning("[%s] TEST_MODE=true -> synthetic fallback is enabled.", sport.upper())
        if historical.empty:
            historical = generate_sample_data(sport, rows=450)
        daily = load_csv_or_empty(daily_path)
        if daily.empty:
            daily = generate_sample_data(sport, rows=12)
        historical = enrich_with_context_features(historical, sport, ROOT)
        daily = enrich_with_context_features(daily, sport, ROOT)
        return historical, daily

    if historical.empty and not model_artifact_path(sport).exists():
        if sport == "mlb":
            if not hist_path.exists():
                print("[MLB] No historical file found — attempting to build...")
                build_mlb_historical()
                historical = load_csv_or_empty(hist_path)
                if not historical.empty:
                    historical = ensure_required_columns(historical)
                    historical = _standardize_historical_features(historical, sport)

            if historical.empty:
                print("[MLB] ⚠️ Missing CSV — using model artifact instead")
        else:
            print(
                f"[{sport.upper()}] ⚠️ Missing CSV — using model artifact instead"
            )

    if historical.empty:
        print(f"[{sport.upper()}] ⚠️ No historical data — skipping training, using baseline")
        LOGGER.warning(
            "[%s] Historical CSV missing but model artifact is present at %s. Runtime training unavailable; fallback artifact may be used.",
            sport.upper(),
            model_artifact_path(sport),
        )

    historical = enrich_with_context_features(historical, sport, ROOT) if not historical.empty else historical
    daily = enrich_with_context_features(fetch_live_daily_odds(sport, today_only=today_only), sport, ROOT)
    return historical, daily
