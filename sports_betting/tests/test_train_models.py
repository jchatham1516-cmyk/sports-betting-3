from pathlib import Path

import pandas as pd
import pytest

from sports_betting.scripts import data_io, train_models


class DummyModel:
    def __init__(self):
        self.trained = False
        self.saved_path = None

    def train(self, historical_df: pd.DataFrame) -> None:
        self.trained = not historical_df.empty

    def save_artifact(self, path: Path) -> None:
        self.saved_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")


@pytest.fixture
def temp_data_root(tmp_path, monkeypatch):
    root = tmp_path / "sports_betting" / "data"
    (root / "historical").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(data_io, "ROOT", root)
    monkeypatch.setattr(train_models, "historical_file_path", data_io.historical_file_path)
    monkeypatch.setattr(train_models, "model_artifact_path", data_io.model_artifact_path)
    monkeypatch.setattr(train_models, "required_historical_columns", data_io.required_historical_columns)
    return root


def test_train_sport_model_writes_artifact(temp_data_root, monkeypatch):
    hist = data_io.historical_file_path("nba")
    cols = data_io.required_historical_columns("nba")
    pd.DataFrame({col: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for col in cols}).to_csv(hist, index=False)

    dummy = DummyModel()
    monkeypatch.setattr(train_models, "choose_model", lambda sport: dummy)

    train_models.train_sport_model("nba")

    assert dummy.trained is True
    assert dummy.saved_path == data_io.model_artifact_path("nba")
    assert data_io.model_artifact_path("nba").exists()


def test_train_sport_model_requires_historical(temp_data_root):
    with pytest.raises(RuntimeError) as exc:
        train_models.train_sport_model("nba")
    assert "Historical CSV does not exist" in str(exc.value)
