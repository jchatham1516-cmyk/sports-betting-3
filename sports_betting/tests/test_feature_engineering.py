from pathlib import Path

import pandas as pd

from sports_betting.sports.common.feature_engineering import enrich_with_context_features, normalize_status


def test_normalize_status_aliases():
    assert normalize_status("Day To Day") == "day-to-day"
    assert normalize_status("IR") == "ir"


def test_enrich_with_context_features_adds_columns(tmp_path: Path):
    root = tmp_path / "sports_betting" / "data"
    (root / "external").mkdir(parents=True)

    injuries = pd.DataFrame(
        [
            {"team": "HOME", "player": "P1", "status": "out", "impact_rating": 1.8, "is_starter": 1, "is_star": 1, "unit": "offense", "qb": 1},
            {"team": "AWAY", "player": "P2", "status": "questionable", "impact_rating": 1.2, "is_starter": 1, "is_star": 0, "unit": "defense", "qb": 0},
        ]
    )
    injuries.to_csv(root / "external" / "nfl_injuries.csv", index=False)

    eff = pd.DataFrame(
        [
            {"team": "HOME", "epa_per_play": 0.2, "success_rate": 0.48},
            {"team": "AWAY", "epa_per_play": -0.1, "success_rate": 0.41},
        ]
    )
    eff.to_csv(root / "external" / "nfl_efficiency.csv", index=False)

    games = pd.DataFrame(
        [
            {"game_id": "1", "home_team": "HOME", "away_team": "AWAY", "event_date": "2025-01-01T00:00:00Z"},
            {"game_id": "2", "home_team": "HOME", "away_team": "AWAY", "event_date": "2025-01-02T00:00:00Z"},
        ]
    )

    out = enrich_with_context_features(games, "nfl", root)
    assert "injury_impact_diff" in out.columns
    assert "travel_fatigue_diff" in out.columns
    assert "epa_per_play_diff" in out.columns
    assert out.loc[0, "qb_out_flag_home"] == 1
