import pandas as pd

from sports_betting.data.fetch_injuries import compute_injury_impact
from sports_betting.sports.nba.features import build_nba_features


def test_nba_empty_injury_source_sets_degraded_neutral_status():
    games = pd.DataFrame([{"home_team": "Boston Celtics", "away_team": "Los Angeles Lakers"}])
    injuries = pd.DataFrame(columns=["sport", "team", "player", "status"])
    injuries.attrs.update({"rows_parsed": 0, "data_quality_status": "degraded"})

    out = compute_injury_impact(games, injuries)

    assert out.loc[0, "injury_impact_home"] == 0.0
    assert out.loc[0, "injury_impact_away"] == 0.0
    assert out.loc[0, "injury_data_stale_flag"] == 1
    assert out.loc[0, "injury_confidence_score"] == 0.0
    assert out.loc[0, "injury_data_quality_status"] == "degraded"


def test_travel_fatigue_diff_is_float_when_decimal_signal_assigned():
    out = build_nba_features(
        pd.DataFrame(
            [
                {
                    "home_team": "Boston Celtics",
                    "away_team": "Los Angeles Lakers",
                    "travel_fatigue_diff": 0,
                }
            ]
        )
    )
    out.loc[0, "travel_fatigue_diff"] = -0.2

    assert str(out["travel_fatigue_diff"].dtype).startswith("float")
    assert out.loc[0, "travel_fatigue_diff"] == -0.2
