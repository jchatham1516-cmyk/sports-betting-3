import pandas as pd

from main import dedupe_final_bets


def test_final_bet_dedupe_keeps_highest_ev_then_highest_odds():
    bets = pd.DataFrame(
        [
            {
                "sport": "mlb",
                "market": "moneyline",
                "selection": "Los Angeles Angels",
                "home_team": "Los Angeles Angels",
                "away_team": "Seattle Mariners",
                "expected_value": 0.04,
                "odds": 110,
            },
            {
                "sport": "mlb",
                "market": "moneyline",
                "selection": "Los Angeles Angels",
                "home_team": "Los Angeles Angels",
                "away_team": "Seattle Mariners",
                "expected_value": 0.05,
                "odds": 100,
            },
            {
                "sport": "mlb",
                "market": "moneyline",
                "selection": "Seattle Mariners",
                "home_team": "Los Angeles Angels",
                "away_team": "Seattle Mariners",
                "expected_value": 0.03,
                "odds": -120,
            },
            {
                "sport": "mlb",
                "market": "moneyline",
                "selection": "Los Angeles Angels",
                "home_team": "Los Angeles Angels",
                "away_team": "Seattle Mariners",
                "expected_value": 0.05,
                "odds": 115,
            },
        ]
    )

    out = dedupe_final_bets(bets)

    angels = out[out["selection"] == "Los Angeles Angels"]
    assert len(out) == 2
    assert len(angels) == 1
    assert angels.iloc[0]["expected_value"] == 0.05
    assert angels.iloc[0]["odds"] == 115
