import pandas as pd

from sports_betting.sports.mlb import pipeline


def test_default_era_values_do_not_count_as_real_pitcher_coverage(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline,
        "get_probable_pitchers",
        lambda: {
            "Team A": "Real Home",
            "Team B": "Real Away",
            "Team C": "Solo Real",
            "Team D": "Unknown D",
            "Team E": "Unknown E",
            "Team F": "Unknown F",
            "Team G": "Unknown G",
            "Team H": "Unknown H",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_pitcher_era_map",
        lambda pitchers: {
            "real home": 3.10,
            "real away": 3.90,
            "solo real": 2.75,
            "fallback default": 4.20,
        },
    )
    daily = pd.DataFrame(
        [
            {"home_team": "Team A", "away_team": "Team B"},
            {"home_team": "Team C", "away_team": "Team D"},
            {"home_team": "Team E", "away_team": "Team F"},
            {"home_team": "Team G", "away_team": "Team H"},
        ]
    )

    out = pipeline._attach_pitcher_data(daily)
    logs = capsys.readouterr().out

    assert out["pitcher_era_home_is_real"].tolist() == [True, True, False, False]
    assert out["pitcher_era_away_is_real"].tolist() == [True, False, False, False]
    assert out["real_home_era_count"].iloc[0] == 2
    assert out["real_away_era_count"].iloc[0] == 1
    assert out["real_both_era_count"].iloc[0] == 1
    assert out["real_pitcher_coverage_pct"].iloc[0] == 25.0
    assert out["data_quality_status"].iloc[0] == "degraded"
    assert out["data_quality_status"].iloc[0] != "normal"
    assert out["default_era_count"].iloc[0] == 5
    assert "[MLB REAL PITCHER ERA COVERAGE]" in logs
    assert "data quality: degraded" in logs
    assert "[MLB PITCHER COVERAGE]" not in logs


def test_mlb_dedupe_uses_event_date_market_and_selection():
    from sports_betting.sports.mlb.pipeline import _dedupe_mlb_odds_rows

    rows = pd.DataFrame(
        [
            {
                "sport": "mlb",
                "event_date": "2026-05-08T23:00:00Z",
                "home_team": "A Team",
                "away_team": "B Team",
                "market": "moneyline",
                "selection": "A Team",
                "home_odds": -110,
                "away_odds": 100,
            },
            {
                "sport": "mlb",
                "event_date": "2026-05-08T23:00:00Z",
                "home_team": "A Team",
                "away_team": "B Team",
                "market": "moneyline",
                "selection": "A Team",
                "home_odds": -110,
                "away_odds": 100,
            },
            {
                "sport": "mlb",
                "event_date": "2026-05-08T23:00:00Z",
                "home_team": "A Team",
                "away_team": "B Team",
                "market": "moneyline",
                "selection": "B Team",
                "home_odds": -110,
                "away_odds": 100,
            },
        ]
    )

    out = _dedupe_mlb_odds_rows(rows)

    assert len(out) == 2
    assert set(out["selection"]) == {"A Team", "B Team"}
