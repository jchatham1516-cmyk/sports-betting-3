from pathlib import Path

import pandas as pd
import pytest

from sports_betting.scripts import data_io


@pytest.fixture
def temp_data_root(tmp_path, monkeypatch):
    root = tmp_path / "sports_betting" / "data"
    (root / "historical").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(data_io, "ROOT", root)
    return root


def _write_valid_historical(path: Path, sport: str):
    cols = data_io.required_historical_columns(sport)
    sample = {c: [0] for c in cols}
    pd.DataFrame(sample).to_csv(path, index=False)


def test_validate_historical_requirements_reports_missing_files(temp_data_root):
    with pytest.raises(RuntimeError) as exc:
        data_io.validate_historical_requirements(sports=["nba"], allow_model_artifacts=False)
    message = str(exc.value)
    assert "sports_betting/data/historical/nba_historical.csv" in message
    assert "Required historical CSV schema by sport" in message


def test_validate_historical_requirements_accepts_model_artifact(temp_data_root):
    artifact = data_io.model_artifact_path("nba")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"placeholder")
    data_io.validate_historical_requirements(sports=["nba"], allow_model_artifacts=True)


def test_validate_historical_requirements_validates_schema(temp_data_root):
    historical = data_io.historical_file_path("nba")
    historical.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_historical(historical, "nba")

    # Remove one required column to force schema error.
    df = pd.read_csv(historical)
    df = df.drop(columns=["home_win"])
    df.to_csv(historical, index=False)

    with pytest.raises(RuntimeError) as exc:
        data_io.validate_historical_requirements(sports=["nba"], allow_model_artifacts=False)
    assert "missing columns: home_win" in str(exc.value)


def test_validate_model_artifacts_exist_reports_missing(temp_data_root):
    with pytest.raises(RuntimeError) as exc:
        data_io.validate_model_artifacts_exist(sports=["nba"])
    assert "Missing artifacts" in str(exc.value)


def test_validate_model_artifacts_exist_accepts_present_artifact(temp_data_root):
    artifact = data_io.model_artifact_path("nba")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"placeholder")
    data_io.validate_model_artifacts_exist(sports=["nba"])


def test_extract_market_prices_maps_home_and_away_to_matching_outcome_names():
    event = {
        "home_team": "Indiana Pacers",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "demo_book",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Indiana Pacers", "price": -847},
                            {"name": "New York Knicks", "price": 570},
                        ],
                    }
                ],
            }
        ],
    }

    prices = data_io._extract_market_prices(event, "h2h")

    assert prices is not None
    assert prices["home_odds"] == -847
    assert prices["away_odds"] == 570
    assert prices["sportsbook_event_home_team"] == "Indiana Pacers"
    assert prices["sportsbook_event_away_team"] == "New York Knicks"
    assert prices["sportsbook_home_outcome_name"] == "Indiana Pacers"
    assert prices["sportsbook_away_outcome_name"] == "New York Knicks"


def test_load_nba_historical_dataset_raises_for_missing_core_columns(temp_data_root):
    historical = data_io.historical_file_path("nba")
    pd.DataFrame({"date": ["2024-01-01"]}).to_csv(historical, index=False)

    with pytest.raises(RuntimeError) as exc:
        data_io.load_nba_historical_dataset()
    assert "missing core required columns" in str(exc.value)


def test_load_nba_historical_dataset_fills_optional_columns(temp_data_root):
    historical = data_io.historical_file_path("nba")
    base = {
        "home_win": [1],
        "home_cover": [0],
        "over_hit": [1],
        "closing_moneyline_home": [-120],
        "closing_spread_home": [-2.5],
        "closing_total": [228.5],
        "elo_diff": [45],
        "rest_diff": [1],
        "injury_impact_diff": [0.2],
        "net_rating_diff": [3.2],
        "pace_diff": [1.5],
        "top_rotation_eff_diff": [0.3],
    }
    pd.DataFrame(base).to_csv(historical, index=False)

    df = data_io.load_nba_historical_dataset()

    assert "travel_fatigue_diff" in df.columns
    assert float(df.loc[0, "travel_fatigue_diff"]) == 0.0
    assert float(df.loc[0, "closing_spread_home"]) == -2.5
