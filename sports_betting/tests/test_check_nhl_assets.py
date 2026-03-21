from pathlib import Path

import pandas as pd

from sports_betting.scripts import check_nhl_assets


def test_fetch_nhl_injury_data_filters_to_nhl(tmp_path: Path, monkeypatch):
    injury_dir = tmp_path / "data" / "injuries"
    injury_dir.mkdir(parents=True)
    injury_path = injury_dir / "injuries.json"
    pd.DataFrame(
        [
            {"sport": "nhl", "team_name": "Boston Bruins", "player": "A", "status": "out"},
            {"sport": "nba", "team_name": "Boston Celtics", "player": "B", "status": "out"},
        ]
    ).to_json(injury_path, orient="records")

    monkeypatch.setattr(check_nhl_assets, "NHL_INJURIES_PATH", injury_path)
    out = check_nhl_assets.fetch_nhl_injury_data()

    assert len(out) == 1
    assert out.iloc[0]["sport"] == "nhl"


def test_check_nhl_data_returns_false_when_missing(tmp_path: Path, monkeypatch):
    hist_path = tmp_path / "data" / "historical" / "nhl_historical.csv"
    monkeypatch.setattr(check_nhl_assets, "NHL_HISTORICAL_PATH", hist_path)

    assert check_nhl_assets.check_nhl_data() is False
    assert hist_path.exists()


def test_adjust_filter_thresholds_defaults():
    out = check_nhl_assets.adjust_filter_thresholds()

    assert out["min_edge"] == 5e-05
    assert out["min_ev"] == 5e-05
    assert out["min_confidence"] == 0.1
