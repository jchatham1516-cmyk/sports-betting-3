"""NHL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NHLModel(NBAModel):
    """Reuse generic logistic workflow with NHL-specific feature definitions."""

    sport = "nhl"
    WIN_FEATURES = [
        "elo_diff",
        "xgf_diff",
        "xga_diff",
        "special_teams_diff",
        "goalie_strength_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "injury_impact_diff",
        "recent_form_diff",
    ]
    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "pace_sum",
        "xg_total",
        "goalie_total_impact",
        "shooting_regression_signal",
        "save_regression_signal",
        "total_line",
    ]
