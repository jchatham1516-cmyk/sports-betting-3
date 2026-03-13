"""NBA disciplined baseline model."""

from __future__ import annotations

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel


class NBAModel(DisciplinedBaselineModel):
    sport = "nba"
