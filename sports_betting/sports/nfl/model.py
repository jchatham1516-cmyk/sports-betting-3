"""NFL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NFLModel(NBAModel):
    """Reuse shared calibrated Gradient Boosting workflow for NFL."""

    sport = "nfl"
