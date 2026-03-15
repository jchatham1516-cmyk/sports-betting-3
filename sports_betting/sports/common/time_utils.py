"""Helpers for timezone-aware game filtering."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz

EASTERN = pytz.timezone("America/New_York")


def filter_games_for_today(games: list[dict]) -> list[dict]:
    """Keep only games that start today in US Eastern Time."""
    today_et = datetime.now(EASTERN).date()

    filtered: list[dict] = []
    for game in games:
        commence = pd.to_datetime(game["commence_time"], utc=True)
        commence_et = commence.tz_convert(EASTERN)

        if commence_et.date() == today_et:
            filtered.append(game)

    return filtered
