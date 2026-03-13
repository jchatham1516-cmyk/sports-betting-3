"""NHL disciplined baseline model."""

from __future__ import annotations

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel


class NHLModel(DisciplinedBaselineModel):
    sport = "nhl"
    PROBABILITY_BOUNDS = {
        "moneyline": (0.14, 0.86),
        "spread": (0.20, 0.80),
        "total": (0.20, 0.80),
    }
