"""Loader for the NBA historical training dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NBA_HISTORICAL_PATH = Path("sports_betting/data/historical/nba_historical.csv")
REQUIRED_CORE_COLUMNS = [
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
OPTIONAL_NUMERIC_COLUMNS = [
    "travel_fatigue_diff",
    "recent_off_rating_diff",
    "recent_def_rating_diff",
    "recent_total_trend",
    "true_shooting_diff",
    "turnover_rate_diff",
    "opening_moneyline_home",
    "opening_spread_home",
    "opening_total",
    "line_movement_spread",
    "line_movement_total",
    "clv_placeholder",
]


def load_nba_historical_dataset(csv_path: str | Path = NBA_HISTORICAL_PATH) -> pd.DataFrame:
    """Read, validate, and normalize the NBA historical dataset."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"NBA historical dataset not found: {path}")

    df = pd.read_csv(path)

    missing_required = [column for column in REQUIRED_CORE_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"NBA historical dataset missing required columns: {', '.join(missing_required)}")

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0

    for column in set(REQUIRED_CORE_COLUMNS + OPTIONAL_NUMERIC_COLUMNS):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    for target in ("home_win", "home_cover", "over_hit"):
        df[target] = df[target].round().clip(0, 1).astype(int)

    return df


__all__ = [
    "NBA_HISTORICAL_PATH",
    "REQUIRED_CORE_COLUMNS",
    "OPTIONAL_NUMERIC_COLUMNS",
    "load_nba_historical_dataset",
]
