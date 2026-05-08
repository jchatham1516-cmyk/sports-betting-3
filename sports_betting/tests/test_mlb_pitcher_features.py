import pandas as pd
import pytest

from sports_betting.sports.mlb.features import build_mlb_features
from sports_betting.sports.mlb.schema import MLB_REQUIRED_FEATURES


def test_build_mlb_features_computes_pitcher_diff():
    frame = pd.DataFrame(
        [
            {
                "pitcher_era_home": 3.15,
                "pitcher_era_away": 4.05,
                "starter_rating_home": 0.2,
                "starter_rating_away": -0.1,
                "bullpen_rating_home": 0.1,
                "bullpen_rating_away": 0.0,
                "hitting_rating_home": 0.4,
                "hitting_rating_away": 0.2,
            }
        ]
    )
    out = build_mlb_features(frame)
    assert out.loc[0, "pitcher_diff"] == pytest.approx(0.90)


def test_pitcher_diff_is_in_required_runtime_features():
    assert "pitcher_diff" in MLB_REQUIRED_FEATURES


def test_starter_ratings_are_created_from_pitcher_era():
    frame = pd.DataFrame(
        [
            {
                "pitcher_era_home": 2.95,
                "pitcher_era_away": 3.61,
                "starter_rating_home": None,
                "starter_rating_away": 0,
            }
        ]
    )
    out = build_mlb_features(frame)
    assert out.loc[0, "starter_rating_home"] == pytest.approx(70.5)
    assert out.loc[0, "starter_rating_away"] == pytest.approx(63.9)
    assert out.loc[0, "starter_rating_diff"] == pytest.approx(6.6)


def test_pitcher_era_rating_is_clamped():
    frame = pd.DataFrame(
        [
            {"pitcher_era_home": 0.50, "pitcher_era_away": 8.00},
        ]
    )
    out = build_mlb_features(frame)
    assert out.loc[0, "starter_rating_home"] == pytest.approx(90.0)
    assert out.loc[0, "starter_rating_away"] == pytest.approx(40.0)
