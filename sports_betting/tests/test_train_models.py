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
        path.write_text("model-disabled-test", encoding="utf-8")


@pytest.fixture
def temp_data_root(tmp_path, monkeypatch):
    root = tmp_path / "sports_betting" / "data"
    (root / "historical").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    monkeypatch.setattr(data_io, "ROOT", root)
    monkeypatch.setattr(train_models, "model_artifact_path", data_io.model_artifact_path)
    monkeypatch.setattr(train_models, "REPORT_PATH", root / "models" / "nba_model_training_report.txt")
    monkeypatch.setitem(train_models.SPORT_DATASET_LOADERS, "nba", data_io.load_nba_historical_dataset)
    return root


def test_train_sport_model_default_skips_artifact_and_writes_report(temp_data_root, monkeypatch):
    hist = data_io.historical_file_path("nba")
    cols = data_io.required_historical_columns("nba")
    pd.DataFrame({col: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for col in cols}).to_csv(hist, index=False)

    dummy = DummyModel()
    monkeypatch.setattr(train_models, "choose_model", lambda sport: dummy)

    train_models.train_sport_model("nba")

    assert dummy.trained is False
    assert dummy.saved_path is None
    assert not data_io.model_artifact_path("nba").exists()
    assert train_models.REPORT_PATH.exists()


def test_train_sport_model_can_write_artifact_when_enabled(temp_data_root, monkeypatch):
    hist = data_io.historical_file_path("nba")
    cols = data_io.required_historical_columns("nba")
    pd.DataFrame({col: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for col in cols}).to_csv(hist, index=False)

    dummy = DummyModel()
    monkeypatch.setattr(train_models, "choose_model", lambda sport: dummy)

    train_models.train_sport_model("nba", save_model_artifacts=True)

    assert dummy.saved_path is None
    assert data_io.model_artifact_path("nba").exists()


def test_train_sport_model_requires_historical(temp_data_root):
    with pytest.raises(RuntimeError) as exc:
        train_models.train_sport_model("nba")
    assert "Historical CSV does not exist" in str(exc.value)
