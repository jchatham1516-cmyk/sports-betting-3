from datetime import datetime, timedelta

import pandas as pd

from sports_betting.sports.common.time_filters import filter_games_for_today


def test_filter_games_for_today_keeps_only_utc_today_rows():
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)

    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "commence_time": [
                f"{today.isoformat()}T10:00:00Z",
                f"{tomorrow.isoformat()}T10:00:00Z",
            ],
        }
    )

    filtered = filter_games_for_today(df)

    assert filtered["game_id"].tolist() == ["g1"]


def test_filter_games_for_today_returns_input_when_column_missing():
    df = pd.DataFrame({"game_id": ["g1"]})

    filtered = filter_games_for_today(df)

    assert filtered.equals(df)
