"""Data loading helpers for live Odds API ingestion with explicit test-mode fallbacks."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sports_betting.sports.common.feature_engineering import SPORT_EFFICIENCY_FEATURES, add_elo_features, enrich_with_context_features
from sports_betting.sports.common.game_filters import current_sports_day_window, filter_games_window

from sports_betting.scripts.build_nba_historical_dataset import NBA_HISTORICAL_COLUMNS


ROOT = Path("sports_betting/data")
LOGGER = logging.getLogger(__name__)
SUPPORTED_SPORTS = ("nba", "nfl", "nhl")

SPORT_TO_ODDS_API_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
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
]

FEATURE_COLUMNS_BY_SPORT = {
    "nba": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nba"]],
    "nfl": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nfl"]],
    "nhl": list(BASE_FEATURE_COLUMNS) + [f"{f}_diff" for f in SPORT_EFFICIENCY_FEATURES["nhl"]],
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


def _standardize_historical_features(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    out = df.copy()
    out["elo_diff"] = _coalesce_numeric(out, ["elo_diff"])
    out["rest_diff"] = _coalesce_numeric(out, ["rest_diff"])
    out["travel_distance"] = _coalesce_numeric(out, ["travel_distance", "travel_fatigue_diff", "travel_diff"])
    out["offensive_rating_diff"] = _coalesce_numeric(
        out, ["offensive_rating_diff", "off_rating_diff", "epa_per_play_diff", "xgf_diff", "off_rating_diff"]
    )
    out["defensive_rating_diff"] = _coalesce_numeric(
        out, ["defensive_rating_diff", "def_rating_diff", "xga_diff", "pressure_rate_diff"]
    )
    out["net_rating_diff"] = _coalesce_numeric(out, ["net_rating_diff", "success_rate_diff", "special_teams_diff"])
    out["injury_impact"] = _coalesce_numeric(out, ["injury_impact", "injury_impact_diff", "qb_impact_diff", "goalie_strength_diff"])
    out["pace"] = _coalesce_numeric(out, ["pace", "pace_diff", "xg_total", "weather_total_impact"])
    out["home_indicator"] = _coalesce_numeric(out, ["home_indicator"], default=1.0).clip(0, 1)
    out["travel_fatigue_diff"] = _coalesce_numeric(out, ["travel_fatigue_diff"])
    out["injury_impact_home"] = _coalesce_numeric(out, ["injury_impact_home"])
    out["injury_impact_away"] = _coalesce_numeric(out, ["injury_impact_away"])
    out["injury_impact_diff"] = _coalesce_numeric(out, ["injury_impact_diff", "injury_impact"]) 
    out["starter_out_count_home"] = _coalesce_numeric(out, ["starter_out_count_home"])
    out["starter_out_count_away"] = _coalesce_numeric(out, ["starter_out_count_away"])
    out["star_player_out_flag_home"] = _coalesce_numeric(out, ["star_player_out_flag_home"])
    out["star_player_out_flag_away"] = _coalesce_numeric(out, ["star_player_out_flag_away"])
    out["starting_goalie_out_flag_home"] = _coalesce_numeric(out, ["starting_goalie_out_flag_home"])
    out["starting_goalie_out_flag_away"] = _coalesce_numeric(out, ["starting_goalie_out_flag_away"])
    out["qb_out_flag_home"] = _coalesce_numeric(out, ["qb_out_flag_home"])
    out["qb_out_flag_away"] = _coalesce_numeric(out, ["qb_out_flag_away"])
    out["offensive_injury_weight_diff"] = _coalesce_numeric(out, ["offensive_injury_weight_diff"])
    out["defensive_injury_weight_diff"] = _coalesce_numeric(out, ["defensive_injury_weight_diff"])
    out["offensive_injury_weight_home"] = _coalesce_numeric(out, ["offensive_injury_weight_home"])
    out["offensive_injury_weight_away"] = _coalesce_numeric(out, ["offensive_injury_weight_away"])
    out["defensive_injury_weight_home"] = _coalesce_numeric(out, ["defensive_injury_weight_home"])
    out["defensive_injury_weight_away"] = _coalesce_numeric(out, ["defensive_injury_weight_away"])
    out["injury_confidence_score"] = _coalesce_numeric(out, ["injury_confidence_score"], default=0.5)
    out["injury_data_stale_flag"] = _coalesce_numeric(out, ["injury_data_stale_flag"])
    out["market_prob"] = _coalesce_numeric(out, ["market_prob"])
    out["model_prob"] = _coalesce_numeric(out, ["model_prob"])
    out["edge"] = _coalesce_numeric(out, ["edge"])
    out["expected_value"] = _coalesce_numeric(out, ["expected_value"])
    out["line_movement"] = _coalesce_numeric(out, ["line_movement"])
    out["clv_placeholder"] = _coalesce_numeric(out, ["clv_placeholder"])
    out["public_favorite_bias_flag"] = _coalesce_numeric(out, ["public_favorite_bias_flag", "public_bias_flag"])
    out["public_bias_flag"] = _coalesce_numeric(out, ["public_bias_flag", "public_favorite_bias_flag"])
    out["favorite_inflation_flag"] = _coalesce_numeric(out, ["favorite_inflation_flag"])
    out["underdog_inflation_flag"] = _coalesce_numeric(out, ["underdog_inflation_flag"])
    out["rest_days_home"] = _coalesce_numeric(out, ["rest_days_home"])
    out["rest_days_away"] = _coalesce_numeric(out, ["rest_days_away"])
    out["back_to_back_home"] = _coalesce_numeric(out, ["back_to_back_home"])
    out["back_to_back_away"] = _coalesce_numeric(out, ["back_to_back_away"])
    out["three_games_in_four_home"] = _coalesce_numeric(out, ["three_games_in_four_home", "three_in_four_home"])
    out["three_games_in_four_away"] = _coalesce_numeric(out, ["three_games_in_four_away", "three_in_four_away"])
    out["three_games_in_four"] = _coalesce_numeric(out, ["three_games_in_four", "three_in_four"])
    out["road_trip_length_home"] = _coalesce_numeric(out, ["road_trip_length_home"])
    out["road_trip_length_away"] = _coalesce_numeric(out, ["road_trip_length_away"])
    out["timezone_shift_home"] = _coalesce_numeric(out, ["timezone_shift_home"])
    out["timezone_shift_away"] = _coalesce_numeric(out, ["timezone_shift_away"])
    out["time_zone_shift"] = _coalesce_numeric(out, ["time_zone_shift", "timezone_shift_away"])
    out["travel_distance_home"] = _coalesce_numeric(out, ["travel_distance_home"])
    out["travel_distance_away"] = _coalesce_numeric(out, ["travel_distance_away"])
    out["rest_days"] = _coalesce_numeric(out, ["rest_days", "rest_days_away"])
    out["back_to_back"] = _coalesce_numeric(out, ["back_to_back", "back_to_back_away"])
    out["starter_out_count"] = _coalesce_numeric(out, ["starter_out_count", "starter_out_count_away"])
    out["star_player_out_flag"] = _coalesce_numeric(out, ["star_player_out_flag", "star_player_out_flag_away"])
    out["qb_out_flag"] = _coalesce_numeric(out, ["qb_out_flag", "qb_out_flag_away"])
    out["starting_goalie_out_flag"] = _coalesce_numeric(out, ["starting_goalie_out_flag", "starting_goalie_out_flag_away"])
    out["last5_net_rating_diff"] = _coalesce_numeric(out, ["last5_net_rating_diff"])
    out["last10_net_rating_diff"] = _coalesce_numeric(out, ["last10_net_rating_diff"])
    out["recent_goal_diff"] = _coalesce_numeric(out, ["recent_goal_diff"])
    out["recent_epa_diff"] = _coalesce_numeric(out, ["recent_epa_diff"])
    out["spread_line"] = _coalesce_numeric(out, ["spread_line", "closing_spread_home"])
    out["total_line"] = _coalesce_numeric(out, ["total_line", "closing_total"])
    out["home_score"] = _coalesce_numeric(out, ["home_score"])
    out["away_score"] = _coalesce_numeric(out, ["away_score"])
    out["date"] = pd.to_datetime(out.get("date", out.get("event_date")), errors="coerce", utc=True)
    out["event_date"] = pd.to_datetime(out.get("event_date", out.get("date")), errors="coerce", utc=True)
    out["closing_moneyline_home"] = _coalesce_numeric(out, ["closing_moneyline_home", "home_odds"])
    out["closing_moneyline_away"] = _coalesce_numeric(out, ["closing_moneyline_away", "away_odds"])
    out["closing_spread_home"] = _coalesce_numeric(out, ["closing_spread_home", "spread_line"])
    out["closing_total"] = _coalesce_numeric(out, ["closing_total", "total_line"])
    out["opening_line"] = _coalesce_numeric(out, ["opening_line", "open_line", "spread_line"])
    out["bet_line"] = _coalesce_numeric(out, ["bet_line", "spread_line"])
    out["current_line"] = _coalesce_numeric(out, ["current_line", "spread_line"])
    out["closing_line"] = _coalesce_numeric(out, ["closing_line", "spread_line"])
    out["clv_diff"] = out["closing_line"] - out["bet_line"]

    # Labels.
    out["home_win"] = _coalesce_numeric(out, ["home_win"]).round().clip(0, 1).astype(int)
    out["home_cover"] = _coalesce_numeric(out, ["home_cover"]).round().clip(0, 1).astype(int)
    out["over_hit"] = _coalesce_numeric(out, ["over_hit"]).round().clip(0, 1).astype(int)

    out = add_elo_features(out, sport)
    LOGGER.info("[%s] Standardized historical feature columns for training.", sport.upper())
    return out


def load_historical_dataset(sport: str) -> pd.DataFrame:
    hist_path = historical_file_path(sport)
    if not hist_path.exists():
        raise RuntimeError(f"[{sport.upper()}] Historical CSV does not exist: {hist_path}")
    return _standardize_historical_features(pd.read_csv(hist_path), sport)


def load_nba_historical_dataset() -> pd.DataFrame:
    hist_path = historical_file_path("nba")
    if not hist_path.exists():
        raise RuntimeError(f"[NBA] Historical CSV does not exist: {hist_path}")

    df = pd.read_csv(hist_path)
    missing_core = [c for c in NBA_CORE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_core:
        raise RuntimeError(
            "[NBA] Historical dataset missing core required columns: " + ", ".join(missing_core)
        )

    # Keep the full NBA schema stable even when optional sources are absent.
    for col in NBA_HISTORICAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    for col in FEATURE_COLUMNS_BY_SPORT["nba"]:
        if col not in df.columns:
            df[col] = 0.0

    standardized = _standardize_historical_features(df, "nba")
    for col in NBA_HISTORICAL_COLUMNS:
        if col in {"date", "season", "game_id", "home_team", "away_team"}:
            continue
        standardized[col] = pd.to_numeric(standardized.get(col, 0.0), errors="coerce").fillna(0.0)
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
    missing_messages: list[str] = []
    schema_messages: list[str] = []

    for sport in sports_to_check:
        hist_path = historical_file_path(sport)
        artifact_path = model_artifact_path(sport)
        hist_exists = hist_path.exists()
        artifact_exists = artifact_path.exists()

        if not hist_exists and not (allow_model_artifacts and artifact_exists):
            missing_messages.append(
                f"- {sport.upper()}: missing both historical CSV ({hist_path}) and trained model artifact ({artifact_path})."
            )
            continue

        if validate_schema and hist_exists:
            raw_df = pd.read_csv(hist_path, nrows=5)
            required_raw = ["home_win", "home_cover", "over_hit", "spread_line", "total_line"]
            missing_raw = [col for col in required_raw if col not in raw_df.columns]
            df = _standardize_historical_features(raw_df, sport)
            required_cols = required_historical_columns(sport)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_raw or missing_cols:
                schema_messages.append(
                    f"- {sport.upper()}: {hist_path} is missing columns: {', '.join(sorted(set(missing_raw + missing_cols)))}"
                )

    if missing_messages or schema_messages:
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
        if missing_messages:
            details.extend(["", "Missing data/model files:", *missing_messages])
        if schema_messages:
            details.extend(["", "Schema issues:", *schema_messages])
        raise RuntimeError("\n".join(details))


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


def _normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    return " ".join(cleaned.split())


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
                return {
                    "sportsbook": book.get("key", "unknown"),
                    "home_odds": int(home_outcome["price"]),
                    "away_odds": int(away_outcome["price"]),
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
        print(f"[{sport.upper()}] No games available today — skipping.")
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

    historical = load_csv_or_empty(hist_path)
    if not historical.empty:
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
        raise RuntimeError(
            f"[{sport.upper()}] Missing historical file at {hist_path}. "
            "Live mode does not auto-generate synthetic training data and no trained model artifact was found."
        )

    if historical.empty:
        LOGGER.warning(
            "[%s] Historical CSV missing but model artifact is present at %s. Skipping in-run retraining.",
            sport.upper(),
            model_artifact_path(sport),
        )

    historical = enrich_with_context_features(historical, sport, ROOT) if not historical.empty else historical
    daily = enrich_with_context_features(fetch_live_daily_odds(sport, today_only=today_only), sport, ROOT)
    return historical, daily
