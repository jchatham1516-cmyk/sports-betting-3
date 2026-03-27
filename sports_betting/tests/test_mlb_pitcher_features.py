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
