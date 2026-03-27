"""NFL live feature enrichment and diff helpers."""

from __future__ import annotations

import pandas as pd

NFL_SOURCE_COLUMNS = [
    "epa_per_play_home",
    "epa_per_play_away",
    "success_rate_home",
    "success_rate_away",
    "qb_efficiency_home",
    "qb_efficiency_away",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def enrich_nfl_live_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if "qb_efficiency_home" not in out.columns:
        out["qb_efficiency_home"] = _num(out, "qb_efficiency_metric_home")
    if "qb_efficiency_away" not in out.columns:
        out["qb_efficiency_away"] = _num(out, "qb_efficiency_metric_away")

    for col in NFL_SOURCE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def build_nfl_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["epa_per_play_diff"] = _num(out, "epa_per_play_home") - _num(out, "epa_per_play_away")
    out["success_rate_diff"] = _num(out, "success_rate_home") - _num(out, "success_rate_away")
    out["qb_efficiency_diff"] = _num(out, "qb_efficiency_home") - _num(out, "qb_efficiency_away")
    return out
