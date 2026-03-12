import math

from sports_betting.sports.common.odds import (
    american_to_decimal,
    american_to_implied_probability,
    expected_value,
    remove_vig_two_way,
)
from sports_betting.sports.common.risk import BankrollConfig, kelly_fraction, recommend_units
from sports_betting.sports.common.selection import confidence_tier


def test_odds_conversions():
    assert math.isclose(american_to_decimal(-110), 1.9090909, rel_tol=1e-5)
    assert math.isclose(american_to_decimal(+150), 2.5, rel_tol=1e-8)
    assert math.isclose(american_to_implied_probability(-110), 0.5238095, rel_tol=1e-5)


def test_no_vig_two_way():
    a, b = remove_vig_two_way(0.55, 0.50)
    assert math.isclose(a + b, 1.0, rel_tol=1e-8)


def test_expected_value_positive_edge():
    ev = expected_value(0.56, -110)
    assert ev > 0


def test_kelly_and_stake_caps():
    cfg = BankrollConfig(bankroll=5000, unit_size=50, max_units_per_bet=1.5, kelly_fraction=0.25)
    kf = kelly_fraction(0.57, -110)
    assert kf > 0
    units = recommend_units(0.57, -110, confidence=0.8, cfg=cfg, mode="fractional_kelly")
    assert 0 <= units <= 1.5


def test_confidence_tier():
    assert confidence_tier(0.85) == "A"
    assert confidence_tier(0.72) == "B"
