import pandas as pd
import pytest

from main import enforce_unique_market_bets, process_predictions, process_predictions_with_adjusted_injury


def test_process_predictions_preserves_edge_ev_and_tracks_injury_columns():
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
    assert out.loc[0, "edge"] == pytest.approx(0.02)
    assert out.loc[0, "expected_value"] == pytest.approx(0.03)
    assert out.loc[0, "injury_impact_home_post"] == pytest.approx(0.2)
    assert out.loc[0, "injury_impact_away_post"] == pytest.approx(0.1)
    assert out.loc[0, "injury_impact_diff_post"] == pytest.approx(0.1)


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
    # mean of all 3 = 0.86666..., threshold = 1.733333...
    assert out["confidence"].max() == pytest.approx(1.7333333333333334)


def test_process_predictions_with_adjusted_injury_alias():
    predictions = pd.DataFrame(
        [
            {"home_team": "atlanta hawks", "away_team": "miami heat", "edge": 0.0, "expected_value": 0.0, "confidence": 0.7}
        ]
    )
    injury_data = pd.DataFrame(
        [
            {"team": "atlanta hawks", "impact": 0.2},
            {"team": "miami heat", "impact": 0.1},
        ]
    )

    out = process_predictions_with_adjusted_injury(predictions, injury_data)
    assert out.loc[0, "edge"] == pytest.approx(0.0)
    assert out.loc[0, "injury_impact_diff_post"] == pytest.approx(0.1)


def test_enforce_unique_market_bets_keeps_single_best_side_per_game_market():
    df = pd.DataFrame(
        [
            {"game_id": "G1", "market": "total", "selection": "Over 245.5", "expected_value": 0.07},
            {"game_id": "G1", "market": "total", "selection": "Under 245.5", "expected_value": 0.03},
            {"game_id": "G1", "market": "spread", "selection": "Home -4.5", "expected_value": 0.05},
            {"game_id": "G1", "market": "spread", "selection": "Away +4.5", "expected_value": 0.04},
            {"game_id": "G2", "market": "total", "selection": "Over 221.5", "expected_value": 0.019},
        ]
    )

    out = enforce_unique_market_bets(df)

    assert len(out) == 2
    assert set(out["selection"]) == {"Over 245.5", "Home -4.5"}
    assert out["expected_value"].min() > 0.02
