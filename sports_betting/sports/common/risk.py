"""Bankroll and staking logic."""

from __future__ import annotations

from dataclasses import dataclass

from sports_betting.sports.common.odds import american_to_decimal


@dataclass
class BankrollConfig:
    bankroll: float
    unit_size: float
    max_units_per_bet: float
    kelly_fraction: float = 0.25


def kelly_fraction(probability: float, american_odds: int) -> float:
    dec = american_to_decimal(american_odds)
    b = dec - 1
    q = 1 - probability
    raw = ((b * probability) - q) / b
    return max(0.0, raw)


def recommend_units(
    probability: float,
    odds: int,
    confidence: float,
    cfg: BankrollConfig,
    mode: str = "fractional_kelly",
) -> float:
    if mode == "flat":
        units = 1.0
    else:
        raw_kelly = kelly_fraction(probability, odds) * cfg.kelly_fraction
        units = (raw_kelly * cfg.bankroll) / cfg.unit_size
    uncertainty_discount = max(0.5, confidence)
    units *= uncertainty_discount
    return round(min(cfg.max_units_per_bet, max(0.0, units)), 2)
