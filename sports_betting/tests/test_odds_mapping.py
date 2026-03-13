from main import _build_game_candidate_bets, _validate_exported_bets_against_sportsbook


def test_knicks_pacers_moneyline_mapping_stays_with_team():
    game_row = {
        "home_team": "Indiana Pacers",
        "away_team": "New York Knicks",
        "home_odds": -847,
        "away_odds": +570,
        "home_spread_odds": -110,
        "away_spread_odds": -110,
        "over_odds": -110,
        "under_odds": -110,
        "total_line": 221.5,
        "spread_line": -11.5,
    }
    predictions = [
        {"market": "moneyline", "side": "Indiana Pacers", "model_probability": 0.88},
        {"market": "moneyline", "side": "New York Knicks", "model_probability": 0.12},
        {"market": "spread", "side": "Indiana Pacers -11.5", "model_probability": 0.52},
        {"market": "spread", "side": "New York Knicks +11.5", "model_probability": 0.48},
        {"market": "total", "side": "Over 221.5", "model_probability": 0.51},
        {"market": "total", "side": "Under 221.5", "model_probability": 0.49},
    ]

    bets = _build_game_candidate_bets(predictions, game_row, "nba", "New York Knicks @ Indiana Pacers")

    pacers_ml = next(b for b in bets if b["market"] == "moneyline" and b["selection"] == "Indiana Pacers")
    knicks_ml = next(b for b in bets if b["market"] == "moneyline" and b["selection"] == "New York Knicks")

    assert pacers_ml["odds"] == -847
    assert knicks_ml["odds"] == 570

    is_valid, warning = _validate_exported_bets_against_sportsbook("game_1", game_row, bets)
    assert is_valid, warning


def test_validate_exported_bets_warns_and_fails_on_moneyline_odds_mismatch():
    game_row = {
        "home_team": "Indiana Pacers",
        "away_team": "New York Knicks",
        "home_odds": -847,
        "away_odds": 570,
        "sportsbook_event_home_team": "Indiana Pacers",
        "sportsbook_event_away_team": "New York Knicks",
    }
    bad_bets = [
        {"market": "moneyline", "selection": "Indiana Pacers", "odds": 570},
        {"market": "moneyline", "selection": "New York Knicks", "odds": -847},
    ]

    is_valid, warning = _validate_exported_bets_against_sportsbook("game_1", game_row, bad_bets)

    assert not is_valid
    assert "odds mismatch" in str(warning)
