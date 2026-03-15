"""Helpers for filtering schedule data by calendar date."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def filter_games_for_today(df: pd.DataFrame) -> pd.DataFrame:
    """Filter games so that only rows scheduled for today (UTC) remain."""
    if "commence_time" not in df.columns:
        return df

    filtered = df.copy()
    filtered["commence_time"] = pd.to_datetime(filtered["commence_time"], utc=True, errors="coerce")

    today = datetime.utcnow().date()
    filtered = filtered[filtered["commence_time"].dt.date == today]

    return filtered
