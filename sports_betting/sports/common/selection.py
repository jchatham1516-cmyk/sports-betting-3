"""Bet qualification and ranking logic."""

from __future__ import annotations

from datetime import date

from sports_betting.models.entities import BetRecommendation, Prediction
from sports_betting.sports.common.risk import BankrollConfig, recommend_units


def confidence_tier(score: float) -> str:
    if score >= 0.82:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.58:
        return "C"
    return "D"


def rank_score(pred: Prediction, model_quality_weight: float = 0.65) -> float:
    return (
        0.40 * pred.expected_value
        + 0.30 * pred.edge
        + 0.20 * pred.confidence
        + 0.10 * model_quality_weight
    )


def qualify_prediction(
    pred: Prediction,
    game_text: str,
    event_date: date,
    odds: int,
    line: float | None,
    thresholds: dict,
    bankroll_cfg: BankrollConfig,
    stake_mode: str,
) -> BetRecommendation | None:
    if pred.edge < thresholds["min_edge"]:
        return None
    if pred.expected_value < thresholds["min_ev"]:
        return None
    if pred.confidence < thresholds["min_confidence"]:
        return None

    stake = recommend_units(pred.model_probability, odds, pred.confidence, bankroll_cfg, mode=stake_mode)
    if stake <= 0:
        return None

    score = rank_score(pred)
    return BetRecommendation(
        event_date=event_date,
        sport=pred.sport,
        game=game_text,
        market=pred.market,
        side=pred.side,
        line=line,
        odds=odds,
        model_probability=pred.model_probability,
        market_implied_probability=pred.market_implied_probability,
        edge=pred.edge,
        expected_value=pred.expected_value,
        confidence_score=pred.confidence,
        confidence_tier=confidence_tier(pred.confidence),
        recommended_units=stake,
        reason_summary=pred.reason_summary,
        flags=pred.flags,
        rank_score=score,
    )
