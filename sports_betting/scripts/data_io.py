"""Data loading helpers for live Odds API ingestion with explicit test-mode fallbacks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path("sports_betting/data")
LOGGER = logging.getLogger(__name__)
SUPPORTED_SPORTS = ("nba", "nfl", "nhl")

SPORT_TO_ODDS_API_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

FEATURE_COLUMNS_BY_SPORT = {
    "nba": [
        "elo_diff",
        "net_rating_diff",
        "off_rating_diff",
        "def_rating_diff",
        "pace_diff",
        "rest_diff",
        "injury_impact_diff",
        "travel_fatigue_diff",
        "recent_form_diff",
        "pace_sum",
        "off_rating_sum",
        "def_rating_sum",
        "recent_total_trend",
        "injury_total_impact",
    ],
    "nfl": [
        "elo_diff",
        "epa_per_play_diff",
        "success_rate_diff",
        "yards_per_play_diff",
        "qb_impact_diff",
        "pressure_rate_diff",
        "turnover_margin_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "pace_sum",
        "off_efficiency_sum",
        "def_efficiency_sum",
        "weather_total_impact",
        "red_zone_efficiency_sum",
    ],
    "nhl": [
        "elo_diff",
        "xgf_diff",
        "xga_diff",
        "special_teams_diff",
        "goalie_strength_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "injury_impact_diff",
        "recent_form_diff",
        "pace_sum",
        "xg_total",
        "goalie_total_impact",
        "shooting_regression_signal",
        "save_regression_signal",
    ],
}

TARGET_COLUMNS = ["home_win", "home_cover", "over_hit"]


def historical_file_path(sport: str) -> Path:
    return ROOT / "historical" / f"{sport}_historical.csv"


def model_artifact_path(sport: str) -> Path:
    return ROOT / "models" / f"{sport}_model.pkl"


def required_historical_columns(sport: str) -> list[str]:
    if sport not in FEATURE_COLUMNS_BY_SPORT:
        raise ValueError(f"Unsupported sport '{sport}'. Supported sports: {', '.join(sorted(FEATURE_COLUMNS_BY_SPORT))}")
    return sorted(set(FEATURE_COLUMNS_BY_SPORT[sport] + ["spread_line", "total_line"] + TARGET_COLUMNS))


def _normalize_sports(sports: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized = list(sports) if sports else list(SUPPORTED_SPORTS)
    unknown = [sport for sport in normalized if sport not in SUPPORTED_SPORTS]
    if unknown:
        raise ValueError(f"Unsupported sport(s): {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_SPORTS)}")
    return normalized


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
            df = pd.read_csv(hist_path, nrows=5)
            required_cols = required_historical_columns(sport)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                schema_messages.append(
                    f"- {sport.upper()}: {hist_path} is missing columns: {', '.join(missing_cols)}"
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
            "spread_line": rng.normal(-2, 4, rows),
            "total_line": rng.normal(220 if sport == "nba" else 44 if sport == "nfl" else 6.0, 8, rows),
            "away_odds": rng.choice([-130, -120, -110, 100, 115, 130], rows),
            "home_odds": rng.choice([-130, -120, -110, 100, 115, 130], rows),
            "away_spread_odds": rng.choice([-112, -110, -108, 100], rows),
            "home_spread_odds": rng.choice([-112, -110, -108, 100], rows),
            "over_odds": rng.choice([-112, -110, -108, 100], rows),
            "under_odds": rng.choice([-112, -110, -108, 100], rows),
            "sportsbook": ["synthetic"] * rows,
        }
    )
    for c in [
        "elo_diff", "net_rating_diff", "off_rating_diff", "def_rating_diff", "pace_diff", "rest_diff", "injury_impact_diff",
        "travel_fatigue_diff", "recent_form_diff", "pace_sum", "off_rating_sum", "def_rating_sum", "recent_total_trend",
        "injury_total_impact", "epa_per_play_diff", "success_rate_diff", "yards_per_play_diff", "qb_impact_diff",
        "pressure_rate_diff", "turnover_margin_diff", "off_efficiency_sum", "def_efficiency_sum", "weather_total_impact",
        "red_zone_efficiency_sum", "xgf_diff", "xga_diff", "special_teams_diff", "goalie_strength_diff", "xg_total",
        "goalie_total_impact", "shooting_regression_signal", "save_regression_signal"
    ]:
        base[c] = rng.normal(0, 1, rows)

    base["home_win"] = (0.08 * base["elo_diff"] + 0.12 * base["rest_diff"] + rng.normal(0, 1, rows) > 0).astype(int)
    base["home_cover"] = (0.1 * base["spread_line"] + 0.08 * base["recent_form_diff"] + rng.normal(0, 1, rows) > 0).astype(int)
    base["over_hit"] = (0.08 * base["pace_sum"] + 0.1 * base["recent_total_trend"] + rng.normal(0, 1, rows) > 0).astype(int)
    return base


def _extract_market_prices(event: dict, market_key: str) -> dict[str, int | float] | None:
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != market_key:
                continue
            outcomes = {out.get("name"): out for out in market.get("outcomes", [])}
            if market_key == "h2h" and {event.get("home_team"), event.get("away_team")} <= set(outcomes.keys()):
                return {
                    "sportsbook": book.get("key", "unknown"),
                    "home_odds": int(outcomes[event["home_team"]]["price"]),
                    "away_odds": int(outcomes[event["away_team"]]["price"]),
                }
            if market_key == "spreads" and {event.get("home_team"), event.get("away_team")} <= set(outcomes.keys()):
                home = outcomes[event["home_team"]]
                away = outcomes[event["away_team"]]
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


def fetch_live_daily_odds(sport: str) -> pd.DataFrame:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required in live mode. Set TEST_MODE=true to use synthetic demo data.")

    odds_sport = SPORT_TO_ODDS_API_KEY.get(sport)
    if not odds_sport:
        raise ValueError(f"Unsupported sport '{sport}'. Supported sports: {', '.join(sorted(SPORT_TO_ODDS_API_KEY))}")

    query = urlencode(
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }
    )
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

    records: list[dict] = []
    for event in payload:
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
            "sportsbook": h2h.get("sportsbook") or spreads.get("sportsbook") or totals.get("sportsbook") or "unknown",
        }
        record.update(h2h)
        record.update(spreads)
        record.update(totals)
        for feature_name in FEATURE_COLUMNS_BY_SPORT[sport]:
            record[feature_name] = 0.0
        records.append(record)

    daily = pd.DataFrame(records)
    if daily.empty:
        print(f"[{sport.upper()}] No games available today — skipping.")
        return None

    daily["event_date"] = pd.to_datetime(daily["event_date"], utc=True, errors="coerce")
    if daily["event_date"].isna().any():
        bad_count = int(daily["event_date"].isna().sum())
        raise RuntimeError(f"[{sport.upper()}] Failed to parse event_date for {bad_count} events from Odds API.")

    LOGGER.info("[%s] Odds API events returned: %s (usable with full markets: %s)", sport.upper(), len(payload), len(daily))
    LOGGER.info("[%s] First teams: %s", sport.upper(), "; ".join((daily["away_team"] + " @ " + daily["home_team"]).head(3).tolist()))
    LOGGER.info("[%s] Sportsbook sources (first 5): %s", sport.upper(), ", ".join(daily["sportsbook"].head(5).tolist()))
    LOGGER.info(
        "[%s] Event dates parsed (first 5): %s",
        sport.upper(),
        ", ".join(daily["event_date"].dt.strftime("%Y-%m-%d %H:%M:%SZ").head(5).tolist()),
    )

    return daily


def load_historical_and_daily(sport: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_path = historical_file_path(sport)
    daily_path = ROOT / "raw" / f"{sport}_daily.csv"

    historical = load_csv_or_empty(hist_path)

    if _is_test_mode():
        LOGGER.warning("[%s] TEST_MODE=true -> synthetic fallback is enabled.", sport.upper())
        if historical.empty:
            historical = generate_sample_data(sport, rows=450)
        daily = load_csv_or_empty(daily_path)
        if daily.empty:
            daily = generate_sample_data(sport, rows=12)
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

    daily = fetch_live_daily_odds(sport)
    return historical, daily
