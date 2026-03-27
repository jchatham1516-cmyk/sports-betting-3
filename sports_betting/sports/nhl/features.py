"""NHL feature engineering shared by training and prediction paths."""

from __future__ import annotations

import pandas as pd

NHL_SOURCE_COLUMNS = [
    "goalie_save_strength_home",
    "goalie_save_strength_away",
    "special_teams_efficiency_home",
    "special_teams_efficiency_away",
    "xgf_home",
    "xgf_away",
    "xga_home",
    "xga_away",
    "shot_share_home",
    "shot_share_away",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def enrich_nhl_live_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "goalie_save_strength_home" not in out.columns:
        out["goalie_save_strength_home"] = _num(out, "goalie_strength_home")
    if "goalie_save_strength_away" not in out.columns:
        out["goalie_save_strength_away"] = _num(out, "goalie_strength_away")

    if "special_teams_efficiency_home" not in out.columns:
        if "pp_home" in out.columns and "pk_home" in out.columns:
            out["special_teams_efficiency_home"] = _num(out, "pp_home") + _num(out, "pk_home")
        else:
            out["special_teams_efficiency_home"] = _num(out, "special_teams_impact_home")
    if "special_teams_efficiency_away" not in out.columns:
        if "pp_away" in out.columns and "pk_away" in out.columns:
            out["special_teams_efficiency_away"] = _num(out, "pp_away") + _num(out, "pk_away")
        else:
            out["special_teams_efficiency_away"] = _num(out, "special_teams_impact_away")

    for col in NHL_SOURCE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    print(
        "[NHL SOURCE DEBUG]",
        out[[
            "goalie_save_strength_home",
            "goalie_save_strength_away",
            "special_teams_efficiency_home",
            "special_teams_efficiency_away",
        ]].head(),
    )
    return out


def build_nhl_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["goalie_diff"] = out["goalie_save_strength_home"] - out["goalie_save_strength_away"]
    out["special_teams_diff"] = out["special_teams_efficiency_home"] - out["special_teams_efficiency_away"]
    out["xgf_diff"] = out["xgf_home"] - out["xgf_away"]
    out["xga_diff"] = out["xga_home"] - out["xga_away"]
    out["shot_share_diff"] = out["shot_share_home"] - out["shot_share_away"]
    out["rest_diff"] = _num(out, "rest_home") - _num(out, "rest_away")
    if "injury_impact_diff" not in out.columns:
        out["injury_impact_diff"] = _num(out, "injury_impact_home") - _num(out, "injury_impact_away")
    else:
        out["injury_impact_diff"] = _num(out, "injury_impact_diff")
    return out


def build_nhl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper used by training code paths."""
    out = enrich_nhl_live_features(df)
    out = build_nhl_diff_features(out)
    out.fillna(0, inplace=True)
    return out
