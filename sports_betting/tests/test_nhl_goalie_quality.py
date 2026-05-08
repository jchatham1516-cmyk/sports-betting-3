import pandas as pd
import pytest

from sports_betting.scripts import feature_enrichment
from sports_betting.scripts.feature_enrichment import _normalize_goalie_team_key
from sports_betting.sports.nhl.features import build_nhl_diff_features


def test_nhl_team_normalization_handles_accents_and_punctuation():
    assert _normalize_goalie_team_key("Montréal Canadiens") == "montreal canadiens"
    assert _normalize_goalie_team_key("St. Louis Blues") == "st louis blues"
    assert _normalize_goalie_team_key("Buffalo") == "buffalo sabres"


def test_nhl_goalie_team_fallback_counts_as_coverage(monkeypatch):
    games = pd.DataFrame(
        [
            {
                "home_team": "Buffalo Sabres",
                "away_team": "Anaheim Ducks",
                "home_odds": -110,
                "away_odds": 100,
            }
        ]
    )
    monkeypatch.setattr(feature_enrichment, "_resolve_nhl_team_stats", lambda: (None, "test"))
    monkeypatch.setattr(
        feature_enrichment,
        "load_nhl_goalies",
        lambda: {
            "buffalo sabres": {"save_pct": 0.912, "source": "nhl_api_club_stats"},
            "anaheim ducks": {"save_pct": 0.899, "source": "nhl_api_club_stats"},
        },
    )

    out = feature_enrichment.enrich_daily_features_by_sport(games, "nhl")

    assert out.loc[0, "goalie_save_home"] == pytest.approx(0.912)
    assert out.loc[0, "goalie_save_away"] == pytest.approx(0.899)
    assert out.loc[0, "goalie_coverage_pct"] == pytest.approx(100.0)
    assert out.loc[0, "goalie_data_quality_status"] == "normal"


def test_nhl_diff_features_add_required_goalie_columns_for_neutral_fallback():
    out = build_nhl_diff_features(pd.DataFrame([{"home_team": "A", "away_team": "B"}]))

    for column in [
        "goalie_save_home",
        "goalie_save_away",
        "goalie_diff",
        "goalie_save_strength_home",
        "goalie_save_strength_away",
        "starting_goalie_out_flag_home",
        "starting_goalie_out_flag_away",
        "goalie_coverage_pct",
        "goalie_data_quality_status",
    ]:
        assert column in out.columns
    assert out.loc[0, "goalie_data_quality_status"] == "severe"


def test_nhl_duplicate_goalie_assignment_is_invalidated(monkeypatch):
    games = pd.DataFrame(
        [
            {
                "home_team": "Buffalo Sabres",
                "away_team": "Montreal Canadiens",
                "home_odds": -110,
                "away_odds": 100,
            }
        ]
    )
    monkeypatch.setattr(feature_enrichment, "_resolve_nhl_team_stats", lambda: (None, "test"))
    monkeypatch.setattr(
        feature_enrichment,
        "load_nhl_goalies",
        lambda: {
            "buffalo sabres": {"save_pct": 0.912, "goalie": "A. Lyon", "source": "nhl_api_probable_goalie"},
            "montreal canadiens": {"save_pct": 0.901, "goalie": "A. Lyon", "source": "nhl_api_probable_goalie"},
        },
    )

    out = feature_enrichment.enrich_daily_features_by_sport(games, "nhl")

    assert out.loc[0, "goalie_save_home"] == pytest.approx(feature_enrichment.NHL_NEUTRAL_GOALIE_SAVE_PCT)
    assert out.loc[0, "goalie_save_away"] == pytest.approx(feature_enrichment.NHL_NEUTRAL_GOALIE_SAVE_PCT)
    assert out.loc[0, "goalie_home"] == ""
    assert out.loc[0, "goalie_away"] == ""
    assert out.loc[0, "goalie_home_source"] == "neutral_invalid"
    assert out.loc[0, "goalie_away_source"] == "neutral_invalid"
    assert out.loc[0, "goalie_coverage_pct"] == pytest.approx(0.0)
    assert out.loc[0, "goalie_data_quality_status"] == "severe"
