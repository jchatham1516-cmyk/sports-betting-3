"""Domain entities for the betting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class GameOdds:
    """Represents one market line for a game."""

    game_id: str
    sport: str
    event_date: date
    away_team: str
    home_team: str
    market: str
    line: float | None
    away_odds: int
    home_odds: int
    book: str = "consensus"


@dataclass
class Prediction:
    """Model probability output for one side of one market."""

    game_id: str
    sport: str
    market: str
    side: str
    model_probability: float
    market_implied_probability: float
    edge: float
    expected_value: float
    confidence: float
    reason_summary: str
    flags: list[str] = field(default_factory=list)
    market_prob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BetRecommendation:
    """Final bet recommendation object after gating and ranking."""

    event_date: date
    sport: str
    game: str
    market: str
    side: str
    line: float | None
    odds: int
    model_probability: float
    market_implied_probability: float
    edge: float
    expected_value: float
    confidence_score: float
    confidence_tier: str
    recommended_units: float
    reason_summary: str
    flags: list[str]
    rank_score: float
    market_prob: float = 0.0
    model_prob: float = 0.0
    line_movement: float = 0.0
    clv_placeholder: float = 0.0
    injury_confidence_score: float = 0.0
    opening_line: float = 0.0
    bet_line: float = 0.0
    current_line: float = 0.0
    closing_line: float = 0.0
    clv_diff: float = 0.0
    fallback_pick: bool = False
