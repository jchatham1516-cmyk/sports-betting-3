"""NFL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NFLModel(NBAModel):
    """Reuse generic logistic workflow with NFL-specific feature definitions."""

    sport = "nfl"
    WIN_FEATURES = [
        "elo_diff",
        "epa_per_play_diff",
        "success_rate_diff",
        "yards_per_play_diff",
        "qb_impact_diff",
        "pressure_rate_diff",
        "turnover_margin_diff",
        "rest_diff",
        "travel_fatigue_diff",
    ]
    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "pace_sum",
        "off_efficiency_sum",
        "def_efficiency_sum",
        "weather_total_impact",
        "red_zone_efficiency_sum",
        "total_line",
    ]
