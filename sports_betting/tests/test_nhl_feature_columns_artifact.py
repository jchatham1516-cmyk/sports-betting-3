from pathlib import Path
import pickle

from sports_betting.scripts.create_placeholder_models import build_payload
from sports_betting.sports.nhl.model import NHLModel


def test_placeholder_payload_includes_feature_columns_for_all_markets():
    payload = build_payload("nhl", NHLModel)

    assert "feature_columns" in payload
    assert payload["feature_columns"]["moneyline"] == NHLModel.WIN_FEATURES
    assert payload["feature_columns"]["spread"] == NHLModel.SPREAD_FEATURES
    assert payload["feature_columns"]["total"] == NHLModel.TOTAL_FEATURES
    assert payload["moneyline_model"].feature_columns == NHLModel.WIN_FEATURES


def test_load_artifact_raises_when_moneyline_feature_columns_missing(tmp_path: Path):
    model = NHLModel()
    artifact = {
        "sport": "nhl",
        "moneyline_model": None,
        "spread_model": None,
        "total_model": None,
        "metrics": {},
        "feature_importance": {},
        "feature_columns": None,
    }
    artifact_path = tmp_path / "nhl_model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    try:
        model.load_artifact(artifact_path)
        assert False, "Expected missing feature columns to raise"
    except ValueError as exc:
        assert str(exc) == "[NHL] Missing stored feature columns for moneyline"
