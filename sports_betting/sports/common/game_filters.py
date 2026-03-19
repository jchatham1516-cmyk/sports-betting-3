"""Helpers for filtering raw Odds API games into the current rolling slate window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


def _games_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the UTC rolling window [now, now + 18h]."""
    start = now
    end = start + timedelta(hours=18)
    return start, end


def filter_games_window(raw_games: list[dict]) -> list[dict]:
    """Keep only games starting within the next 18 hours."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    cutoff = now + timedelta(hours=18)

    filtered: list[dict] = []
    for game in raw_games:
        commence_raw = game.get("commence_time")
        if not commence_raw:
            continue

        commence = pd.to_datetime(commence_raw, utc=True, errors="coerce")
        if pd.isna(commence):
            continue

        if now <= commence <= cutoff:
            filtered.append(game)

    return filtered


def filter_games_for_today(raw_games: list[dict]) -> list[dict]:
    """Backward-compatible alias for the rolling 18-hour game window filter."""
    return filter_games_window(raw_games)


def current_sports_day_window() -> tuple[datetime, datetime]:
    """Expose current UTC rolling bounds for query parameter construction."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return _games_window(now)
