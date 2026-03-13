"""NHL model implementation."""

from __future__ import annotations

from sports_betting.sports.nba.model import NBAModel


class NHLModel(NBAModel):
    """Reuse shared calibrated Gradient Boosting workflow for NHL."""

    sport = "nhl"
    BASE_FEATURES = NBAModel.BASE_FEATURES + [
        "xgf_diff",
        "xga_diff",
        "xgf_pct_diff",
        "shot_share_diff",
        "special_teams_efficiency_diff",
        "goalie_save_strength_diff",
        "goalie_xgsaved_proxy_diff",
        "top_line_impact_diff",
        "defensive_pair_impact_diff",
    ]
    WIN_FEATURES = BASE_FEATURES
    SPREAD_FEATURES = BASE_FEATURES + ["spread_line"]
    TOTAL_FEATURES = BASE_FEATURES + ["total_line"]
