"""NHL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NHLModel(NBAModel):
    """Reuse shared calibrated Gradient Boosting workflow for NHL."""

    sport = "nhl"
