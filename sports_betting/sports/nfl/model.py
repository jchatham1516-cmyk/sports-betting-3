"""NFL disciplined baseline model."""

from __future__ import annotations

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel


class NFLModel(DisciplinedBaselineModel):
    sport = "nfl"
    PROBABILITY_BOUNDS = {
        "moneyline": (0.15, 0.85),
        "spread": (0.20, 0.80),
        "total": (0.20, 0.80),
    }
