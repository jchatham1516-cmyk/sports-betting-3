"""Soccer 3-way candidate generation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports_betting.sports.common.odds import expected_value

from .model import predict_outcome_probabilities, train_soccer_model


def _vig_free_market_probs(home_odds: int, draw_odds: int, away_odds: int) -> tuple[float, float, float]:
    home_imp = 1 / (1 + (home_odds / 100.0)) if home_odds > 0 else abs(home_odds) / (abs(home_odds) + 100)
    draw_imp = 1 / (1 + (draw_odds / 100.0)) if draw_odds > 0 else abs(draw_odds) / (abs(draw_odds) + 100)
    away_imp = 1 / (1 + (away_odds / 100.0)) if away_odds > 0 else abs(away_odds) / (abs(away_odds) + 100)
    total = home_imp + draw_imp + away_imp
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return home_imp / total, draw_imp / total, away_imp / total


def run_soccer_pipeline(historical_df: pd.DataFrame, daily_df: pd.DataFrame) -> list[dict]:
    model_bundle = train_soccer_model(historical_df)
    if model_bundle is None or daily_df.empty:
        return []

    frame = daily_df.copy()
    probs = predict_outcome_probabilities(model_bundle, frame)
    frame = frame.join(probs)

    candidates: list[dict] = []
    for _, row in frame.iterrows():
        home_odds = int(row.get("home_odds", 0))
        draw_odds = int(row.get("draw_odds", 0))
        away_odds = int(row.get("away_odds", 0))
        if home_odds == 0 or draw_odds == 0 or away_odds == 0:
            continue

        market_home, market_draw, market_away = _vig_free_market_probs(home_odds, draw_odds, away_odds)
        game = f"{row.get('away_team')} @ {row.get('home_team')}"
        common = {
            "sport": "soccer",
            "event_id": str(row.get("game_id", "")),
            "commence_time": row.get("commence_time", row.get("event_date")),
            "away_team": str(row.get("away_team", "")),
            "home_team": str(row.get("home_team", "")),
            "game": game,
            "market": "moneyline",
            "market_type": "moneyline_3way",
            "home_odds": home_odds,
            "away_odds": away_odds,
            "support_count": 0,
            "composite_score": 0.0,
            "reason_summary": "Soccer multinomial 3-way moneyline prediction",
            "injury_impact_home": float(row.get("injury_impact_home", 0.0)),
            "injury_impact_away": float(row.get("injury_impact_away", 0.0)),
            "injury_impact_diff": float(row.get("injury_impact_diff", 0.0)),
        }

        selections = [
            (str(row.get("home_team", "")), home_odds, float(row["home_prob"]), market_home),
            ("draw", draw_odds, float(row["draw_prob"]), market_draw),
            (str(row.get("away_team", "")), away_odds, float(row["away_prob"]), market_away),
        ]

        for selection, odds, model_probability, market_probability in selections:
            edge = model_probability - market_probability
            candidates.append(
                {
                    **common,
                    "selection": selection,
                    "odds": int(odds),
                    "model_prob": model_probability,
                    "model_probability": model_probability,
                    "market_probability": float(market_probability),
                    "edge": edge,
                    "expected_value": expected_value(model_probability, int(odds)),
                    "confidence": float(np.clip(0.45 + edge, 0.01, 0.99)),
                }
            )

    return candidates
