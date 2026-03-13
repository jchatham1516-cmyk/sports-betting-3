"""NHL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NHLModel(NBAModel):
    """Reuse calibrated ensemble workflow with NHL-specific features."""

    sport = "nhl"
    WIN_FEATURES = [
        "elo_diff",
        "xgf_diff",
        "xga_diff",
        "xg_total",
        "goalie_strength_diff",
        "special_teams_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "recent_form_diff",
        "shooting_regression_signal",
        "save_regression_signal",
        "market_spread",
        "market_total",
    ]
    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "xgf_diff",
        "xga_diff",
        "xg_total",
        "goalie_strength_diff",
        "special_teams_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "recent_form_diff",
        "shooting_regression_signal",
        "save_regression_signal",
        "market_spread",
        "market_total",
        "total_line",
    ]
