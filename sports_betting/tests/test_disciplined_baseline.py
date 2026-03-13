import numpy as np
import pandas as pd

from sports_betting.sports.nba.model import NBAModel


def _make_training_frame(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 80, rows),
            "rest_diff": rng.normal(0, 1.2, rows),
            "travel_fatigue_diff": rng.normal(0, 0.4, rows),
            "travel_distance": rng.uniform(0, 1800, rows),
            "injury_impact_diff": rng.normal(0, 1.0, rows),
            "offensive_rating_diff": rng.normal(0, 4.0, rows),
            "defensive_rating_diff": rng.normal(0, 4.0, rows),
            "net_rating_diff": rng.normal(0, 5.0, rows),
            "pace_diff": rng.normal(0, 3.0, rows),
            "pace": rng.normal(197, 7, rows),
            "spread_line": rng.normal(0, 6.5, rows),
            "total_line": rng.normal(225, 9, rows),
        }
    )
    score = (
        0.012 * df["elo_diff"]
        + 0.25 * df["rest_diff"]
        - 0.2 * df["travel_fatigue_diff"]
        + 0.35 * df["net_rating_diff"]
        - 0.28 * df["injury_impact_diff"]
    )
    p = 1 / (1 + np.exp(-score / 3.0))
    df["home_win"] = (rng.uniform(0, 1, rows) < p).astype(int)
    df["home_cover"] = (rng.uniform(0, 1, rows) < np.clip(p + 0.02, 0.05, 0.95)).astype(int)
    df["over_hit"] = (rng.uniform(0, 1, rows) < np.clip(0.5 + 0.001 * (df["pace"] - 197), 0.05, 0.95)).astype(int)
    return df


def test_disciplined_model_predictions_include_support_signals():
    model = NBAModel()
    train_df = _make_training_frame()
    model.train(train_df)

    daily = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_odds": -120,
                "away_odds": 110,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
                "spread_line": -3.5,
                "total_line": 227.5,
                "elo_diff": 72.0,
                "rest_diff": 1.0,
                "travel_fatigue_diff": 0.6,
                "travel_distance": 980.0,
                "injury_impact_diff": -1.1,
                "offensive_rating_diff": 3.1,
                "defensive_rating_diff": -1.0,
                "net_rating_diff": 4.0,
                "pace_diff": 1.7,
                "pace": 201.0,
            }
        ]
    )

    preds = model.predict_daily(daily)
    assert len(preds) == 6
    for pred in preds:
        assert 0.0 <= pred.model_probability <= 1.0
        assert "support_count" in pred.metadata
        assert "strength" in pred.reason_summary
