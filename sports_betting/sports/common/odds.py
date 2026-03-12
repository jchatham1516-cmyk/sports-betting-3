"""Odds conversion and probability utilities."""

from __future__ import annotations


def american_to_decimal(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero.")
    if american_odds > 0:
        return 1 + (american_odds / 100)
    return 1 + (100 / abs(american_odds))


def american_to_implied_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Implied probability sum must be > 0.")
    return prob_a / total, prob_b / total


def expected_value(probability: float, american_odds: int, stake: float = 1.0) -> float:
    """Return EV in stake units."""
    decimal_odds = american_to_decimal(american_odds)
    win_profit = stake * (decimal_odds - 1)
    loss = stake
    return probability * win_profit - (1 - probability) * loss
