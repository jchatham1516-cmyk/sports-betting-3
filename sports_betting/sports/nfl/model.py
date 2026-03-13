"""NFL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NFLModel(NBAModel):
    """Reuse calibrated ensemble workflow with NFL-specific features."""

    sport = "nfl"
    WIN_FEATURES = [
        "elo_diff",
        "epa_per_play_diff",
        "success_rate_diff",
        "yards_per_play_diff",
        "turnover_margin_diff",
        "pressure_rate_diff",
        "qb_impact_diff",
        "weather_total_impact",
        "rest_diff",
        "travel_fatigue_diff",
        "market_spread",
        "market_total",
    ]
    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "epa_per_play_diff",
        "success_rate_diff",
        "yards_per_play_diff",
        "weather_total_impact",
        "rest_diff",
        "travel_fatigue_diff",
        "market_spread",
        "market_total",
        "total_line",
    ]
