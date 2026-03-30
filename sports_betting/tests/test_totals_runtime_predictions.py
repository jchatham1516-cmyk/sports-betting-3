import numpy as np
import pandas as pd

from main import _build_runtime_moneyline_predictions, train_runtime_totals_model


def test_runtime_totals_predictions_are_data_driven():
    rows = 120
    rng = np.random.default_rng(42)
    historical = pd.DataFrame(
        {
            "home_score": rng.integers(90, 130, rows),
            "away_score": rng.integers(88, 126, rows),
            "total_line": rng.normal(220, 9, rows),
            "offensive_rating_home": rng.normal(114, 4, rows),
            "offensive_rating_away": rng.normal(113, 4, rows),
            "defensive_rating_home": rng.normal(111, 4, rows),
            "defensive_rating_away": rng.normal(111, 4, rows),
            "pace_home": rng.normal(100, 2, rows),
            "pace_away": rng.normal(100, 2, rows),
            "pace_diff": rng.normal(0, 2, rows),
            "true_shooting_home": rng.normal(0.58, 0.02, rows),
            "true_shooting_away": rng.normal(0.57, 0.02, rows),
            "effective_fg_home": rng.normal(0.55, 0.02, rows),
            "effective_fg_away": rng.normal(0.54, 0.02, rows),
            "turnover_rate_home": rng.normal(0.13, 0.01, rows),
            "turnover_rate_away": rng.normal(0.13, 0.01, rows),
            "rebound_rate_home": rng.normal(0.50, 0.02, rows),
            "rebound_rate_away": rng.normal(0.50, 0.02, rows),
            "free_throw_rate_home": rng.normal(0.20, 0.02, rows),
            "free_throw_rate_away": rng.normal(0.20, 0.02, rows),
            "injury_impact_home": rng.normal(0, 0.4, rows),
            "injury_impact_away": rng.normal(0, 0.4, rows),
            "home_win": rng.integers(0, 2, rows),
        }
    )

    totals_model = train_runtime_totals_model(historical)
    assert totals_model is not None

    daily = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_odds": -130,
                "away_odds": 110,
                "over_odds": -105,
                "under_odds": -115,
                "total_line": 224.5,
                "offensive_rating_home": 118.0,
                "offensive_rating_away": 116.5,
                "defensive_rating_home": 112.0,
                "defensive_rating_away": 113.5,
                "pace_home": 101.0,
                "pace_away": 100.5,
                "pace_diff": 0.5,
                "true_shooting_home": 0.59,
                "true_shooting_away": 0.58,
                "effective_fg_home": 0.56,
                "effective_fg_away": 0.55,
                "turnover_rate_home": 0.125,
                "turnover_rate_away": 0.130,
                "rebound_rate_home": 0.51,
                "rebound_rate_away": 0.49,
                "free_throw_rate_home": 0.21,
                "free_throw_rate_away": 0.20,
                "injury_impact_home": 0.1,
                "injury_impact_away": 0.0,
            }
        ]
    )

    class DummyRuntimeModel:
        feature_columns = ["home_odds"]

        def predict_proba(self, x):
            return np.array([[0.45, 0.55]])

    runtime_preds = _build_runtime_moneyline_predictions(daily, DummyRuntimeModel(), totals_model, "nba")
    totals = [p for p in runtime_preds if p.market == "total"]
    assert len(totals) == 2
    over = next(p for p in totals if p.side.startswith("Over"))
    under = next(p for p in totals if p.side.startswith("Under"))
    assert over.model_probability != 0.5
    assert under.model_probability != 0.5
    assert abs((over.model_probability + under.model_probability) - 1.0) < 1e-9
