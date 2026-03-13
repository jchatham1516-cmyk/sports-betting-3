from pathlib import Path

import pandas as pd

from sports_betting.scripts import build_nba_historical_dataset as builder


def test_build_nba_historical_dataset_writes_expected_schema(tmp_path, monkeypatch):
    root = tmp_path / "sports_betting" / "data"
    (root / "historical").mkdir(parents=True)
    (root / "raw").mkdir(parents=True)
    monkeypatch.setattr(builder, "ROOT", root)

    out = builder.build_nba_historical_dataset(root / "raw")

    assert out == root / "historical" / "nba_historical.csv"
    assert out.exists()

    df = pd.read_csv(out)
    assert set(builder.NBA_HISTORICAL_COLUMNS).issubset(df.columns)
    assert {"home_win", "home_cover", "over_hit"}.issubset(df.columns)
    assert len(df) > 0
