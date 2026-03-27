"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

import pandas as pd


def enrich_daily_features_by_sport(df: pd.DataFrame, sport_name: str) -> pd.DataFrame:
    sport = str(sport_name).lower()

    if sport == "nba":
        from sports_betting.sports.nba.features import enrich_nba_live_features, build_nba_diff_features

        df = enrich_nba_live_features(df)
        df = build_nba_diff_features(df)
        return df

    if sport == "nhl":
        from sports_betting.sports.nhl.features import enrich_nhl_live_features, build_nhl_diff_features

        df = enrich_nhl_live_features(df)
        df = build_nhl_diff_features(df)
        return df

    if sport == "mlb":
        from sports_betting.sports.mlb.features import enrich_mlb_live_features, build_mlb_features

        df = enrich_mlb_live_features(df)
        df = build_mlb_features(df)
        return df

    if sport == "nfl":
        from sports_betting.sports.nfl.features import enrich_nfl_live_features, build_nfl_diff_features

        df = enrich_nfl_live_features(df)
        df = build_nfl_diff_features(df)
        return df

    if sport == "soccer":
        from sports_betting.sports.soccer.features import enrich_soccer_live_features, build_soccer_features

        df = enrich_soccer_live_features(df)
        df = build_soccer_features(df)
        return df

    return df


def validate_feature_signal(df: pd.DataFrame, sport_name: str) -> None:
    sport = str(sport_name).lower()

    checks = {
        "nba": ["offensive_rating_diff", "defensive_rating_diff", "net_rating_diff"],
        "nhl": ["goalie_diff", "special_teams_diff", "xgf_diff"],
        "mlb": ["starter_rating_diff", "bullpen_rating_diff", "hitting_rating_diff"],
        "nfl": ["epa_per_play_diff", "success_rate_diff", "qb_efficiency_diff"],
    }

    cols = checks.get(sport, [])
    if not cols:
        return

    existing = [c for c in cols if c in df.columns]
    if not existing:
        print(f"[{sport.upper()} VALIDATION] No feature-signal columns found.")
        return

    zero_like: list[str] = []
    for col in existing:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if series.abs().sum() == 0:
            zero_like.append(col)

    if zero_like:
        print(f"[{sport.upper()} VALIDATION WARNING] Zero-signal columns: {zero_like}")
