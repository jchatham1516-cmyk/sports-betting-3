"""MLB candidate generation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports_betting.sports.common.odds import american_to_implied_probability, expected_value, remove_vig_two_way

from .model import predict_home_probability, train_mlb_model


def run_mlb_pipeline(historical_df: pd.DataFrame, daily_df: pd.DataFrame) -> list[dict]:
    model_bundle = train_mlb_model(historical_df)
    if model_bundle is None or daily_df.empty:
        return []

    frame = daily_df.copy()
    frame["home_prob"] = predict_home_probability(model_bundle, frame).clip(0.01, 0.99)
    frame["away_prob"] = (1.0 - frame["home_prob"]).clip(0.01, 0.99)

    candidates: list[dict] = []
    for _, row in frame.iterrows():
        home_odds = int(row.get("home_odds", 0))
        away_odds = int(row.get("away_odds", 0))
        if home_odds == 0 or away_odds == 0:
            continue

        home_market_raw = american_to_implied_probability(home_odds)
        away_market_raw = american_to_implied_probability(away_odds)
        market_home, market_away = remove_vig_two_way(home_market_raw, away_market_raw)

        game = f"{row.get('away_team')} @ {row.get('home_team')}"
        common = {
            "sport": "mlb",
            "event_id": str(row.get("game_id", "")),
            "commence_time": row.get("commence_time", row.get("event_date")),
            "away_team": str(row.get("away_team", "")),
            "home_team": str(row.get("home_team", "")),
            "game": game,
            "market": "moneyline",
            "market_type": "moneyline",
            "support_count": 0,
            "composite_score": 0.0,
            "reason_summary": "MLB runtime logistic moneyline prediction",
            "injury_impact_home": float(row.get("injury_impact_home", 0.0)),
            "injury_impact_away": float(row.get("injury_impact_away", 0.0)),
            "injury_impact_diff": float(row.get("injury_impact_diff", 0.0)),
        }

        selections = [
            (str(row.get("home_team", "")), home_odds, float(row["home_prob"]), float(market_home)),
            (str(row.get("away_team", "")), away_odds, float(row["away_prob"]), float(market_away)),
        ]
        for selection, odds, model_probability, market_probability in selections:
            edge = model_probability - market_probability
            candidates.append(
                {
                    **common,
                    "selection": selection,
                    "odds": int(odds),
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "model_prob": model_probability if selection == str(row.get("home_team", "")) else 1.0 - model_probability,
                    "model_probability": model_probability,
                    "market_probability": market_probability,
                    "edge": edge,
                    "expected_value": expected_value(model_probability, int(odds)),
                    "confidence": float(np.clip(0.5 + edge, 0.01, 0.99)),
                }
            )

    return candidates
