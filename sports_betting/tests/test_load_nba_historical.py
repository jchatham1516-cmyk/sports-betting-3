import pandas as pd
import pytest

from sports_betting.scripts.load_nba_historical import load_nba_historical_dataset


def test_loader_validates_required_columns_and_fills_optional(tmp_path):
    csv_path = tmp_path / "nba_historical.csv"
    df = pd.DataFrame(
        {
            "home_win": [1, 0],
            "home_cover": [1, 0],
            "over_hit": [0, 1],
            "closing_moneyline_home": [-120, 110],
            "closing_spread_home": [-2.5, 2.5],
            "closing_total": [228.5, 224.0],
            "elo_diff": [15, -12],
            "rest_diff": [1, -1],
            "injury_impact_diff": [0.3, -0.2],
            "net_rating_diff": [4.5, -3.1],
            "pace_diff": [1.2, -0.5],
            "top_rotation_eff_diff": [2.0, -1.0],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_nba_historical_dataset(csv_path)

    assert "travel_fatigue_diff" in loaded.columns
    assert loaded["travel_fatigue_diff"].tolist() == [0.0, 0.0]


def test_loader_raises_when_required_missing(tmp_path):
    csv_path = tmp_path / "nba_historical.csv"
    pd.DataFrame({"home_win": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as exc:
        load_nba_historical_dataset(csv_path)

    assert "missing required columns" in str(exc.value)
