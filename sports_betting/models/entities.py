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
