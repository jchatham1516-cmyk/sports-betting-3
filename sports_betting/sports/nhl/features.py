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

NHL_REQUIRED_SOURCE_COLUMNS = [
    "goalie_save_strength_home",
    "goalie_save_strength_away",
    "special_teams_efficiency_home",
    "special_teams_efficiency_away",
    "xgf_home",
    "xgf_away",
    "xga_home",
    "xga_away",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _merge_nhl_team_stats(df: pd.DataFrame, nhl_team_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stats = nhl_team_stats.copy()
    if "team" not in stats.columns:
        raise RuntimeError("[DATA ERROR] nhl_team_stats is missing required column: team")

    if "home_team_norm" not in out.columns or "away_team_norm" not in out.columns:
        if "home_team" not in out.columns or "away_team" not in out.columns:
            raise RuntimeError("[DATA ERROR] Missing team columns required for NHL stat merge")
        out["home_team_norm"] = out["home_team"].astype(str).str.lower().str.strip()
        out["away_team_norm"] = out["away_team"].astype(str).str.lower().str.strip()

    stats["team"] = stats["team"].astype(str).str.lower().str.strip()
    out = out.merge(
        stats,
        left_on="home_team_norm",
        right_on="team",
        how="left",
    )
    out = out.merge(
        stats,
        left_on="away_team_norm",
        right_on="team",
        how="left",
        suffixes=("_home", "_away"),
    )
    return out


def enrich_nhl_live_features(df: pd.DataFrame, nhl_team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()

    if nhl_team_stats is not None:
        out = _merge_nhl_team_stats(out, nhl_team_stats)

    missing = [col for col in NHL_REQUIRED_SOURCE_COLUMNS if col not in out.columns]
    if missing:
        raise RuntimeError(f"[DATA ERROR] Missing required source stats: {missing}")

    for col in NHL_SOURCE_COLUMNS:
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
