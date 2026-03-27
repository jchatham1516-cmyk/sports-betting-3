"""NHL feature engineering shared by training and prediction paths."""

from __future__ import annotations

import pandas as pd


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_nhl_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Goalie diff
    if "goalie_save_strength_home" in out.columns and "goalie_save_strength_away" in out.columns:
        out["goalie_diff"] = _num(out, "goalie_save_strength_home") - _num(out, "goalie_save_strength_away")
    elif "goalie_strength_home" in out.columns and "goalie_strength_away" in out.columns:
        out["goalie_diff"] = _num(out, "goalie_strength_home") - _num(out, "goalie_strength_away")
    else:
        out["goalie_diff"] = 0.0
    out["goalie_diff"] = pd.to_numeric(out["goalie_diff"], errors="coerce").fillna(0.0)

    out["xgf_diff"] = _num(out, "xgf_home") - _num(out, "xgf_away")
    out["xga_diff"] = _num(out, "xga_home") - _num(out, "xga_away")
    out["shot_share_diff"] = _num(out, "shot_share_home") - _num(out, "shot_share_away")

    # Special teams diff
    if "special_teams_efficiency_home" in out.columns and "special_teams_efficiency_away" in out.columns:
        out["special_teams_diff"] = _num(out, "special_teams_efficiency_home") - _num(out, "special_teams_efficiency_away")
    elif "pp_home" in out.columns and "pk_away" in out.columns:
        out["special_teams_diff"] = _num(out, "pp_home") - _num(out, "pk_away")
    else:
        out["special_teams_diff"] = 0.0
    out["special_teams_diff"] = pd.to_numeric(out["special_teams_diff"], errors="coerce").fillna(0.0)

    out["rest_diff"] = _num(out, "rest_home") - _num(out, "rest_away")
    if "injury_impact_diff" not in out.columns:
        out["injury_impact_diff"] = _num(out, "injury_impact_home") - _num(out, "injury_impact_away")
    else:
        out["injury_impact_diff"] = _num(out, "injury_impact_diff")

    out.fillna(0, inplace=True)
    return out
