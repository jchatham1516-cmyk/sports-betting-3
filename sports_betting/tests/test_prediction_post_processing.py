import pandas as pd
import pytest

from main import process_predictions


def test_process_predictions_amplifies_injury_impact():
    predictions = pd.DataFrame(
        [
            {
                "home_team": "atlanta hawks",
                "away_team": "miami heat",
                "edge": 0.02,
                "expected_value": 0.03,
                "confidence": 0.8,
            }
        ]
    )
    injury_data = pd.DataFrame(
        [
            {"team": "atlanta hawks", "impact": 0.2},
            {"team": "miami heat", "impact": 0.1},
        ]
    )

    out = process_predictions(predictions, injury_data)
    # delta=(0.2-0.1)*5 => +0.5
    assert out.loc[0, "edge"] == pytest.approx(0.52)
    assert out.loc[0, "expected_value"] == pytest.approx(0.53)


def test_process_predictions_caps_confidence_by_recent_window():
    predictions = pd.DataFrame(
        [
            {"home_team": "a", "away_team": "b", "edge": 0.0, "expected_value": 0.0, "confidence": 0.2},
            {"home_team": "a", "away_team": "b", "edge": 0.0, "expected_value": 0.0, "confidence": 0.4},
            {"home_team": "a", "away_team": "b", "edge": 0.0, "expected_value": 0.0, "confidence": 2.0},
        ]
    )
    injury_data = pd.DataFrame(columns=["team", "impact"])
    out = process_predictions(predictions, injury_data)
    # mean of all 3 = 0.86666..., threshold = 1.3
    assert out["confidence"].max() == pytest.approx(1.3)
