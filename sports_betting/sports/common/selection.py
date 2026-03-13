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
    return 0.40 * pred.expected_value + 0.30 * pred.edge + 0.20 * pred.confidence + 0.10 * model_quality_weight


def _has_severe_data_issue(pred: Prediction) -> bool:
    severe_flags = {"stale_injury_data", "missing_key_features", "suspicious_placeholder", "extreme_probability_without_support"}
    return bool(severe_flags.intersection(set(pred.flags)))


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
    if _has_severe_data_issue(pred):
        return None
    if pred.model_probability < 0.52:
        return None
    if odds > 2000:
        return None
    min_edge = float(thresholds.get("min_edge", 0.02))
    if pred.edge < min_edge:
        return None
    if pred.expected_value <= 0 or pred.expected_value < thresholds["min_ev"]:
        return None
    if pred.confidence < thresholds["min_confidence"]:
        return None

    stake = recommend_units(pred.model_probability, odds, pred.confidence, bankroll_cfg, mode=stake_mode, edge=pred.edge)
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
        market_prob=pred.market_implied_probability,
        model_prob=pred.model_probability,
        line_movement=float(pred.metadata.get("line_movement", 0.0)),
        clv_placeholder=float(pred.metadata.get("clv_placeholder", 0.0)),
        injury_confidence_score=float(pred.metadata.get("injury_confidence_score", 0.0)),
        opening_line=float(pred.metadata.get("opening_line", pred.metadata.get("current_line", 0.0))),
        bet_line=float(pred.metadata.get("bet_line", pred.metadata.get("current_line", 0.0))),
        current_line=float(pred.metadata.get("current_line", 0.0)),
        closing_line=float(pred.metadata.get("closing_line", pred.metadata.get("current_line", 0.0))),
        clv_diff=float(pred.metadata.get("clv_diff", 0.0)),
    )
