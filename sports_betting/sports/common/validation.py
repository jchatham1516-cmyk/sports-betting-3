"""Validation helpers."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "moneyline": ["away_odds", "home_odds"],
    "spread": ["spread_line", "away_spread_odds", "home_spread_odds"],
    "total": ["total_line", "over_odds", "under_odds"],
}


def validate_required_columns(df: pd.DataFrame, market: str) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS.get(market, []) if c not in df.columns]
    return missing
