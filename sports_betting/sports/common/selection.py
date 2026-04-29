"""Bet qualification and ranking logic."""

from __future__ import annotations

from datetime import date
import logging
import math

from sports_betting.models.entities import BetRecommendation, Prediction
from sports_betting.sports.common.risk import BankrollConfig, recommend_units

LOGGER = logging.getLogger(__name__)


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
    severe_flags = {"missing_key_features", "suspicious_placeholder", "extreme_probability_without_support"}
    return bool(severe_flags.intersection(set(pred.flags)))


def _passes_base_checks(pred: Prediction, odds: int) -> bool:
    if _has_severe_data_issue(pred):
        return False
    if pred.model_probability < 0.52:
        return False
    if odds > 2000:
        return False
    if pred.model_probability < 0.55 or pred.model_probability > 0.75:
        return False
    return True


def _build_recommendation(
    pred: Prediction,
    game_text: str,
    event_date: date,
    odds: int,
    line: float | None,
    bankroll_cfg: BankrollConfig,
    stake_mode: str,
) -> BetRecommendation | None:
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


def candidate_prediction(
    pred: Prediction,
    game_text: str,
    event_date: date,
    odds: int,
    line: float | None,
    bankroll_cfg: BankrollConfig,
    stake_mode: str,
) -> BetRecommendation | None:
    if not _passes_base_checks(pred, odds):
        return None

    return _build_recommendation(pred, game_text, event_date, odds, line, bankroll_cfg, stake_mode)


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
    rec = candidate_prediction(pred, game_text, event_date, odds, line, bankroll_cfg, stake_mode)
    if rec is None:
        return None

    min_ev = float(thresholds.get("min_ev", 0.00001))
    edge = 0.0 if math.isnan(rec.edge) else rec.edge
    expected_value = min_ev if math.isnan(rec.expected_value) else rec.expected_value
    confidence = rec.confidence_score
    base_units = rec.recommended_units
    rec.edge = edge
    rec.expected_value = expected_value

    pass_filter = False

    # SOFT EDGE PENALTY (no hard rejection)
    if edge < 0:
        confidence *= 0.8

    # OPTIONAL SAFETY FLOOR
    if expected_value < -0.05:
        pass_filter = False

    # SHARPNESS FILTER
    if edge > 0.03 and expected_value > 0.05 and 0.55 <= rec.model_probability <= 0.75:
        pass_filter = True

    confidence_multiplier = max(0.5, min(1.2, confidence))
    adjusted_units = base_units * confidence_multiplier
    rec.recommended_units = adjusted_units
    rec.confidence_score = confidence
    rec.confidence_tier = confidence_tier(confidence)

    print("[FINAL FILTER DEBUG]")
    print("EV:", expected_value)
    print("Confidence:", confidence)
    print("Units:", adjusted_units)

    if not pass_filter:
        LOGGER.info(
            "[FILTER] rejected %s %s (%s): ev %.4f did not meet EV pass criteria",
            pred.sport,
            game_text,
            pred.market,
            expected_value,
        )
        return None
    return rec
