import math
from datetime import date

from sports_betting.models.entities import Prediction
from sports_betting.sports.common.odds import (
    american_to_decimal,
    american_to_implied_probability,
    expected_value,
    remove_vig_two_way,
)
from sports_betting.sports.common.risk import BankrollConfig, kelly_fraction, recommend_units
from sports_betting.sports.common.selection import confidence_tier, qualify_prediction


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


def test_qualify_prediction_scales_units_instead_of_hard_confidence_reject():
    cfg = BankrollConfig(bankroll=5000, unit_size=50, max_units_per_bet=2.0, kelly_fraction=0.25)
    pred = Prediction(
        game_id="game-1",
        sport="nba",
        market="moneyline",
        side="home",
        model_probability=0.54,
        market_implied_probability=0.50,
        edge=0.04,
        expected_value=0.03,
        confidence=0.02,
        reason_summary="test",
    )

    rec = qualify_prediction(
        pred=pred,
        game_text="Away @ Home",
        event_date=date(2026, 3, 25),
        odds=110,
        line=None,
        thresholds={"min_ev": 0.00001, "min_confidence": 0.95},
        bankroll_cfg=cfg,
        stake_mode="fractional_kelly",
    )

    assert rec is not None
    assert rec.recommended_units > 0
    assert rec.recommended_units <= 1.0


def test_qualify_prediction_extreme_low_confidence_safety_rejects_low_ev():
    cfg = BankrollConfig(bankroll=5000, unit_size=50, max_units_per_bet=2.0, kelly_fraction=0.25)
    pred = Prediction(
        game_id="game-2",
        sport="nba",
        market="moneyline",
        side="away",
        model_probability=0.60,
        market_implied_probability=0.50,
        edge=0.10,
        expected_value=0.03,
        confidence=0.005,
        reason_summary="test",
    )

    rec = qualify_prediction(
        pred=pred,
        game_text="Away @ Home",
        event_date=date(2026, 3, 25),
        odds=120,
        line=None,
        thresholds={"min_ev": 0.00001, "min_confidence": 0.0},
        bankroll_cfg=cfg,
        stake_mode="fractional_kelly",
    )

    assert rec is None
