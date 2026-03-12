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
