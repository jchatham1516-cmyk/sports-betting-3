"""NFL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NFLModel(NBAModel):
    """Reuse shared calibrated Gradient Boosting workflow for NFL."""

    sport = "nfl"
    BASE_FEATURES = NBAModel.BASE_FEATURES + [
        "epa_per_play_home",
        "epa_per_play_away",
        "epa_per_play_diff",
        "success_rate_home",
        "success_rate_away",
        "success_rate_diff",
        "yards_per_play_diff",
        "pressure_rate_diff",
        "sack_rate_diff",
        "explosive_play_rate_diff",
        "qb_impact_home",
        "qb_impact_away",
        "qb_efficiency_metric_diff",
        "offensive_line_grade_proxy_diff",
        "defensive_efficiency_diff",
        "special_teams_impact_diff",
        "weather_total_impact_diff",
    ]
    WIN_FEATURES = BASE_FEATURES
    SPREAD_FEATURES = BASE_FEATURES + ["spread_line"]
    TOTAL_FEATURES = BASE_FEATURES + ["total_line"]
