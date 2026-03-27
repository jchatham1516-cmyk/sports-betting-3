"""MLB feature engineering helpers."""

from __future__ import annotations

import pandas as pd


MLB_FEATURE_COLUMNS = [
    "implied_home_prob",
    "pitcher_era_diff",
    "pitcher_xera_diff",
    "pitcher_whip_diff",
    "pitcher_k_rate_diff",
    "pitcher_last3_diff",
    "bullpen_era_diff",
    "bullpen_fatigue_diff",
    "runs_per_game_diff",
    "ops_diff",
    "slugging_diff",
    "team_k_rate_diff",
    "home_split_diff",
    "last10_diff",
    "rest_diff",
    "injury_impact_diff",
]


def _num(df: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_mlb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["implied_home_prob"] = _num(out, ["implied_home_prob"])
    if "implied_home_prob" not in df.columns and "home_odds" in out.columns:
        home_odds = pd.to_numeric(out["home_odds"], errors="coerce").fillna(0)
        out["implied_home_prob"] = home_odds.apply(
            lambda odds: abs(odds) / (abs(odds) + 100) if odds < 0 else (100 / (odds + 100) if odds > 0 else 0.5)
        )

    out["pitcher_era_diff"] = _num(out, ["pitcher_era_home"]) - _num(out, ["pitcher_era_away"])
    out["pitcher_xera_diff"] = _num(out, ["pitcher_xera_home", "pitcher_xERA_home"]) - _num(out, ["pitcher_xera_away", "pitcher_xERA_away"])
    out["pitcher_whip_diff"] = _num(out, ["pitcher_whip_home"]) - _num(out, ["pitcher_whip_away"])
    out["pitcher_k_rate_diff"] = _num(out, ["pitcher_k_rate_home"]) - _num(out, ["pitcher_k_rate_away"])
    out["pitcher_last3_diff"] = _num(out, ["pitcher_last3_starts_home", "pitcher_recent_form_home"]) - _num(out, ["pitcher_last3_starts_away", "pitcher_recent_form_away"])

    out["bullpen_era_diff"] = _num(out, ["bullpen_era_home"]) - _num(out, ["bullpen_era_away"])
    out["bullpen_fatigue_diff"] = _num(out, ["bullpen_usage_home", "bullpen_fatigue_home"]) - _num(out, ["bullpen_usage_away", "bullpen_fatigue_away"])

    out["runs_per_game_diff"] = _num(out, ["runs_per_game_home"]) - _num(out, ["runs_per_game_away"])
    out["ops_diff"] = _num(out, ["ops_home", "OPS_home"]) - _num(out, ["ops_away", "OPS_away"])
    out["slugging_diff"] = _num(out, ["slugging_home", "slg_home"]) - _num(out, ["slugging_away", "slg_away"])
    out["team_k_rate_diff"] = _num(out, ["team_k_rate_home"]) - _num(out, ["team_k_rate_away"])

    out["home_split_diff"] = _num(out, ["home_split_home", "home_record_strength"]) - _num(out, ["away_split_away", "away_record_strength"])
    out["last10_diff"] = _num(out, ["last10_home", "last10_wins_home"]) - _num(out, ["last10_away", "last10_wins_away"])
    out["rest_diff"] = _num(out, ["rest_days_home"]) - _num(out, ["rest_days_away"])

    if "injury_impact_diff" not in out.columns:
        out["injury_impact_diff"] = _num(out, ["injury_impact_home"]) - _num(out, ["injury_impact_away"])
    else:
        out["injury_impact_diff"] = _num(out, ["injury_impact_diff"])

    return out
