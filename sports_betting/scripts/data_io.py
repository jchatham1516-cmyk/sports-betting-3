"""Data loading helpers with resilient defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("sports_betting/data")


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def generate_sample_data(sport: str, rows: int = 300) -> pd.DataFrame:
    """Generate synthetic feature-rich data to make project runnable out-of-box."""
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


def load_historical_and_daily(sport: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_path = ROOT / "historical" / f"{sport}_historical.csv"
    daily_path = ROOT / "raw" / f"{sport}_daily.csv"

    historical = load_csv_or_empty(hist_path)
    daily = load_csv_or_empty(daily_path)
    if historical.empty:
        historical = generate_sample_data(sport, rows=450)
    if daily.empty:
        daily = generate_sample_data(sport, rows=12)
    return historical, daily
