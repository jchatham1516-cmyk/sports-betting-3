"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_team_stats_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"[DATA ERROR] Missing required team stats file: {path}")
    return pd.read_csv(path)


def enrich_daily_features_by_sport(df: pd.DataFrame, sport_name: str) -> pd.DataFrame:
    sport = str(sport_name).lower()

    if sport == "nba":
        from sports_betting.sports.nba.features import enrich_nba_live_features, build_nba_diff_features

        nba_team_stats = _load_team_stats_csv(Path("sports_betting/data/external/nba_team_stats.csv"))
        df = enrich_nba_live_features(df, nba_team_stats=nba_team_stats)
        df = build_nba_diff_features(df)
        return df

    if sport == "nhl":
        from sports_betting.sports.nhl.features import enrich_nhl_live_features, build_nhl_diff_features

        nhl_team_stats = _load_team_stats_csv(Path("sports_betting/data/external/nhl_team_stats.csv"))
        df = enrich_nhl_live_features(df, nhl_team_stats=nhl_team_stats)
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
    def validate_not_zero(frame: pd.DataFrame, cols: list[str], sport_label: str) -> None:
        for col in cols:
            if col not in frame.columns:
                raise RuntimeError(f"[{sport_label}] Missing required validation feature: {col}")
            series = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
            if series.abs().sum() == 0:
                raise RuntimeError(f"[{sport_label}] Feature {col} is all zero — data pipeline broken")

    sport = str(sport_name).lower()

    checks = {
        "nba": ["offensive_rating_diff", "defensive_rating_diff"],
        "nhl": ["goalie_diff", "special_teams_diff"],
        "mlb": ["starter_rating_diff", "hitting_rating_diff"],
        "nfl": ["epa_per_play_diff", "success_rate_diff", "qb_efficiency_diff"],
    }

    cols = checks.get(sport, [])
    if not cols:
        return

    validate_not_zero(df, cols, sport.upper())
