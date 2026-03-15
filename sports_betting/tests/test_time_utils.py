from datetime import datetime, timedelta

import pandas as pd

from sports_betting.sports.common.time_utils import EASTERN, filter_games_for_today


def test_filter_games_for_today_uses_eastern_calendar_day():
    now_et = datetime.now(EASTERN)
    today_et = now_et.date()
    tomorrow_et = today_et + timedelta(days=1)

    game_today_utc = pd.Timestamp(datetime.combine(today_et, datetime.min.time()), tz=EASTERN).tz_convert("UTC")
    game_tomorrow_utc = pd.Timestamp(datetime.combine(tomorrow_et, datetime.min.time()), tz=EASTERN).tz_convert("UTC")

    games = [
        {"id": "g1", "commence_time": game_today_utc.isoformat()},
        {"id": "g2", "commence_time": game_tomorrow_utc.isoformat()},
    ]

    filtered = filter_games_for_today(games)

    assert [g["id"] for g in filtered] == ["g1"]
