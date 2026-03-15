"""Helpers for filtering raw Odds API games into a sports-day window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


def _sports_day_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the UTC sports-day window [start, end] anchored at 12:00 UTC."""
    start = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now.hour < 12:
        start = start - timedelta(days=1)
    end = start + timedelta(days=1)
    return start, end


def filter_games_for_today(raw_games: list[dict]) -> list[dict]:
    """Keep games occurring between today 12:00 UTC and tomorrow 12:00 UTC."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    start, end = _sports_day_window(now)

    filtered: list[dict] = []
    for game in raw_games:
        commence_raw = game.get("commence_time")
        if not commence_raw:
            continue

        commence = pd.to_datetime(commence_raw, utc=True, errors="coerce")
        if pd.isna(commence):
            continue

        if start <= commence <= end:
            filtered.append(game)

    return filtered


def current_sports_day_window() -> tuple[datetime, datetime]:
    """Expose current UTC sports-day bounds for query parameter construction."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return _sports_day_window(now)
