from datetime import datetime, timedelta

import pandas as pd

from sports_betting.sports.common.final_game_filter import ET, filter_predictions_today


def test_filter_predictions_today_keeps_only_rows_in_et_sports_window():
    start = datetime.now(ET).replace(hour=4, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    in_window = (start + timedelta(hours=1)).astimezone(tz=pd.Timestamp.utcnow().tzinfo)
    out_window = (end + timedelta(hours=1)).astimezone(tz=pd.Timestamp.utcnow().tzinfo)

    df = pd.DataFrame(
        {
            "game_id": ["in", "out"],
            "metadata": [
                {"commence_time": in_window.isoformat()},
                {"commence_time": out_window.isoformat()},
            ],
        }
    )

    filtered = filter_predictions_today(df)

    assert filtered["game_id"].tolist() == ["in"]


def test_filter_predictions_today_returns_input_when_metadata_missing():
    df = pd.DataFrame({"game_id": ["g1"]})

    filtered = filter_predictions_today(df)

    assert filtered.equals(df)
