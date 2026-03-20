"""NBA disciplined baseline model."""

from __future__ import annotations

from sports_betting.sports.common.baseline_model import DisciplinedBaselineModel


class NBAModel(DisciplinedBaselineModel):
    sport = "nba"

    WIN_FEATURES = [
        "implied_home_prob",
        "spread",
        "spread_abs",
        "is_favorite",
        "elo_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "injury_impact_diff",
        "net_rating_diff",
        "top_rotation_eff_diff",
        "recent_off_rating_diff",
        "recent_def_rating_diff",
    ]
    SPREAD_FEATURES = [
        "rest_diff",
        "injury_impact_diff",
        "net_rating_diff",
        "top_rotation_eff_diff",
        "travel_fatigue_diff",
        "closing_spread_home",
    ]
    TOTAL_FEATURES = [
        "pace_diff",
        "recent_total_trend",
        "injury_impact_diff",
        "closing_total",
        "true_shooting_diff",
        "turnover_rate_diff",
    ]
