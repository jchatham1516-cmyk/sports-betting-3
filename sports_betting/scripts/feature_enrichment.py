"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _load_historical_csv(sport_name: str) -> pd.DataFrame | None:
    return _safe_read_csv(Path(f"sports_betting/data/historical/{sport_name}_historical.csv"))


def _normalize_team(name: object) -> str:
    return str(name).lower().strip()


def build_nba_team_stats(historical: pd.DataFrame) -> pd.DataFrame:
    frame = historical.copy()
    frame["home_team_norm"] = frame["home_team"].apply(_normalize_team)
    frame["away_team_norm"] = frame["away_team"].apply(_normalize_team)

    frame["home_score"] = pd.to_numeric(frame.get("home_score", 0.0), errors="coerce")
    frame["away_score"] = pd.to_numeric(frame.get("away_score", 0.0), errors="coerce")
    frame["point_diff"] = frame["home_score"] - frame["away_score"]

    home = (
        frame.groupby("home_team_norm", as_index=False)
        .agg(
            home_score=("home_score", "mean"),
            away_score=("away_score", "mean"),
            point_diff=("point_diff", "mean"),
        )
        .rename(
            columns={
                "home_team_norm": "team_norm",
                "home_score": "points_for",
                "away_score": "points_against",
            }
        )
    )
    away = (
        frame.groupby("away_team_norm", as_index=False)
        .agg(
            away_score=("away_score", "mean"),
            home_score=("home_score", "mean"),
            point_diff=("point_diff", "mean"),
        )
        .rename(
            columns={
                "away_team_norm": "team_norm",
                "away_score": "points_for",
                "home_score": "points_against",
            }
        )
    )

    return pd.concat([home, away], ignore_index=True).groupby("team_norm", as_index=False).mean(numeric_only=True)


def build_nhl_team_stats(historical: pd.DataFrame) -> pd.DataFrame:
    frame = historical.copy()
    frame["home_team_norm"] = frame["home_team"].apply(_normalize_team)
    frame["away_team_norm"] = frame["away_team"].apply(_normalize_team)

    home = frame.groupby("home_team_norm", as_index=False).agg(
        goalie_save_strength=("goalie_strength_home", "mean"),
        xgf_home=("xgf_home", "mean"),
        xga_home=("xga_home", "mean"),
    )
    away = frame.groupby("away_team_norm", as_index=False).agg(
        goalie_save_strength=("goalie_strength_away", "mean"),
        xgf_home=("xgf_away", "mean"),
        xga_home=("xga_away", "mean"),
    )
    away = away.rename(columns={"away_team_norm": "team_norm"})
    home = home.rename(columns={"home_team_norm": "team_norm"})
    return pd.concat([home, away], ignore_index=True).groupby("team_norm", as_index=False).mean(numeric_only=True)


def build_mlb_team_stats(historical: pd.DataFrame) -> pd.DataFrame:
    frame = historical.copy()
    frame["home_team_norm"] = frame["home_team"].apply(_normalize_team)
    frame["away_team_norm"] = frame["away_team"].apply(_normalize_team)

    home = frame.groupby("home_team_norm", as_index=False).agg(
        starter_rating=("starter_rating_home", "mean"),
        bullpen_rating=("bullpen_rating_home", "mean"),
        hitting_rating=("hitting_rating_home", "mean"),
        home_split=("home_split_home", "mean"),
        recent_form=("recent_form_home", "mean"),
    )
    away = frame.groupby("away_team_norm", as_index=False).agg(
        starter_rating=("starter_rating_away", "mean"),
        bullpen_rating=("bullpen_rating_away", "mean"),
        hitting_rating=("hitting_rating_away", "mean"),
        home_split=("home_split_away", "mean"),
        recent_form=("recent_form_away", "mean"),
    )
    away = away.rename(columns={"away_team_norm": "team_norm"})
    home = home.rename(columns={"home_team_norm": "team_norm"})
    return pd.concat([home, away], ignore_index=True).groupby("team_norm", as_index=False).mean(numeric_only=True)


def _resolve_nba_team_stats() -> tuple[pd.DataFrame | None, str]:
    candidates = [
        Path("sports_betting/data/external/nba_team_stats.csv"),
        Path("sports_betting/data/nba_team_stats.csv"),
        Path("sports_betting/data/inputs/nba_team_stats.csv"),
    ]
    for path in candidates:
        data = _safe_read_csv(path)
        if data is not None:
            return data, f"external:{path}"

    historical_df = _load_historical_csv("nba")
    if historical_df is None:
        return None, "missing"
    derived = build_nba_team_stats(historical_df)
    if derived is None:
        return None, "historical_empty"
    return derived, "historical_derived"


def _resolve_nhl_team_stats() -> tuple[pd.DataFrame | None, str]:
    candidates = [
        Path("sports_betting/data/external/nhl_team_stats.csv"),
        Path("sports_betting/data/nhl_team_stats.csv"),
        Path("sports_betting/data/inputs/nhl_team_stats.csv"),
    ]
    for path in candidates:
        data = _safe_read_csv(path)
        if data is not None:
            return data, f"external:{path}"

    historical_df = _load_historical_csv("nhl")
    if historical_df is None:
        return None, "missing"

    required = {"goalie_strength_home", "goalie_strength_away", "xgf_home", "xgf_away", "xga_home", "xga_away"}
    if not required.issubset(historical_df.columns):
        return None, "historical_missing_columns"
    derived = build_nhl_team_stats(historical_df)
    if derived is None:
        return None, "historical_empty"
    return derived, "historical_derived"


def _resolve_mlb_team_stats() -> tuple[pd.DataFrame | None, str]:
    candidates = [
        Path("sports_betting/data/external/mlb_team_stats.csv"),
        Path("sports_betting/data/mlb_team_stats.csv"),
        Path("sports_betting/data/inputs/mlb_team_stats.csv"),
    ]
    for path in candidates:
        data = _safe_read_csv(path)
        if data is not None:
            return data, f"external:{path}"

    historical_df = _load_historical_csv("mlb")
    if historical_df is None:
        return None, "missing"

    required = {"starter_rating_home", "starter_rating_away", "bullpen_rating_home", "bullpen_rating_away"}
    if not required.issubset(historical_df.columns):
        return None, "historical_missing_columns"
    derived = build_mlb_team_stats(historical_df)
    if derived is None:
        return None, "historical_empty"
    return derived, "historical_derived"


def _merge_home_away_team_stats(df: pd.DataFrame, team_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    stats = team_df.copy()
    if "team" in stats.columns and "team_norm" not in stats.columns:
        stats = stats.rename(columns={"team": "team_norm"})
    if "team_norm" not in stats.columns:
        raise RuntimeError("[DATA ERROR] Team stats must include `team` or `team_norm`")
    stats["team_norm"] = stats["team_norm"].apply(_normalize_team)

    if "home_team_norm" not in out.columns and "home_team" in out.columns:
        out["home_team_norm"] = out["home_team"].apply(_normalize_team)
    if "away_team_norm" not in out.columns and "away_team" in out.columns:
        out["away_team_norm"] = out["away_team"].apply(_normalize_team)

    target_columns = set(mapping.values()) | {dst.replace("_home", "_away") for dst in mapping.values()}
    cols_to_drop = [col for col in out.columns if col in target_columns or col.endswith("_x") or col.endswith("_y")]
    out = out.drop(columns=cols_to_drop, errors="ignore")

    resolved_mapping = {
        src: (dst[:-5] if dst.endswith("_home") else dst)
        for src, dst in mapping.items()
        if src in stats.columns
    }
    stats_base = stats.rename(columns=resolved_mapping)
    home_stats = stats_base.add_suffix("_home")
    away_stats = stats_base.add_suffix("_away")

    out = out.merge(home_stats, left_on="home_team_norm", right_on="team_norm_home", how="left")
    out = out.merge(away_stats, left_on="away_team_norm", right_on="team_norm_away", how="left")

    print("[MERGE SUCCESS CHECK]")
    if {"home_team", "offensive_rating_home", "offensive_rating_away"}.issubset(out.columns):
        print(out[["home_team", "offensive_rating_home", "offensive_rating_away"]].head())
    elif {"home_team", "goalie_save_strength_home", "goalie_save_strength_away"}.issubset(out.columns):
        print(out[["home_team", "goalie_save_strength_home", "goalie_save_strength_away"]].head())
    elif {"home_team", "starter_rating_home", "starter_rating_away"}.issubset(out.columns):
        print(out[["home_team", "starter_rating_home", "starter_rating_away"]].head())

    if "home_team_norm" in out.columns and "team_norm" in stats.columns:
        missing = set(out["home_team_norm"]) - set(stats["team_norm"])
        if missing:
            print("[MERGE MISMATCH HOME TEAMS]")
            print(missing)

    for base_col in mapping.values():
        away_col = base_col.replace("_home", "_away")
        if base_col in out.columns:
            out[base_col] = out[base_col]
        if away_col in out.columns:
            out[away_col] = out[away_col]
    return out


def enrich_daily_features_by_sport(df: pd.DataFrame, sport_name: str) -> pd.DataFrame:
    sport = str(sport_name).lower()

    if sport == "nba":
        from sports_betting.sports.nba.features import enrich_nba_live_features, build_nba_diff_features

        team_df, source = _resolve_nba_team_stats()
        print(f"[NBA ENRICHMENT SOURCE] {source}")
        if team_df is not None:
            mapping = {
                "offensive_rating_home": "offensive_rating_home",
                "defensive_rating_home": "defensive_rating_home",
                "net_rating_home": "net_rating_home",
                "pace_home": "pace_home",
                "offensive_rating": "offensive_rating_home",
                "defensive_rating": "defensive_rating_home",
                "net_rating": "net_rating_home",
                "pace": "pace_home",
                "true_shooting": "true_shooting_home",
                "effective_fg": "effective_fg_home",
                "turnover_rate": "turnover_rate_home",
                "rebound_rate": "rebound_rate_home",
                "free_throw_rate": "free_throw_rate_home",
                "points_for": "points_for_home",
                "points_against": "points_against_home",
                "point_diff": "point_diff_home",
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[NBA ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty", "historical_missing_columns"}:
                print("[NBA INFO] Using basic stat fallback (score-based features)")

        if {"points_for_home", "points_against_home", "points_for_away", "points_against_away"}.issubset(df.columns):
            df["point_diff_diff"] = (df["points_for_home"] - df["points_against_home"]) - (
                df["points_for_away"] - df["points_against_away"]
            )
        df = enrich_nba_live_features(df, nba_team_stats=None)
        df = build_nba_diff_features(df)
        return df

    if sport == "nhl":
        from sports_betting.sports.nhl.features import enrich_nhl_live_features, build_nhl_diff_features

        team_df, source = _resolve_nhl_team_stats()
        print(f"[NHL ENRICHMENT SOURCE] {source}")
        if team_df is not None:
            mapping = {
                "goalie_save_strength": "goalie_save_strength_home",
                "goalie_strength": "goalie_save_strength_home",
                "xgf": "xgf_home",
                "xga": "xga_home",
                "xgf_home": "xgf_home",
                "xga_home": "xga_home",
                "special_teams_efficiency": "special_teams_efficiency_home",
                "shot_share": "shot_share_home",
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[NHL ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty", "historical_missing_columns"}:
                raise RuntimeError("[NHL DATA ERROR] No team stat source found after exhausting external and historical fallbacks.")
        df = enrich_nhl_live_features(df, nhl_team_stats=None)
        df = build_nhl_diff_features(df)
        return df

    if sport == "mlb":
        from sports_betting.sports.mlb.features import enrich_mlb_live_features, build_mlb_features

        team_df, source = _resolve_mlb_team_stats()
        print(f"[MLB ENRICHMENT SOURCE] {source}")
        if team_df is not None:
            mapping = {
                "starter_rating": "starter_rating_home",
                "bullpen_rating": "bullpen_rating_home",
                "hitting_rating": "hitting_rating_home",
                "home_split": "home_split_home",
                "recent_form": "recent_form_home",
                "starter_rating_home": "starter_rating_home",
                "bullpen_rating_home": "bullpen_rating_home",
                "hitting_rating_home": "hitting_rating_home",
                "home_split_home": "home_split_home",
                "recent_form_home": "recent_form_home",
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[MLB ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty", "historical_missing_columns"}:
                raise RuntimeError("[MLB DATA ERROR] No team stat source found after exhausting external and historical fallbacks.")
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
                print(f"[{sport_label}] WARNING: Feature {col} is all zero — verify source availability and merges.")

    sport = str(sport_name).lower()

    if sport == "nhl":
        if "goalie_diff" not in df.columns:
            print("[WARNING] NHL goalie_diff missing after enrichment")
            return
        goalie_signal = pd.to_numeric(df["goalie_diff"], errors="coerce").fillna(0.0).abs().sum()
        if goalie_signal == 0:
            print("[WARNING] NHL still has low signal")
        return

    checks = {
        "nba": ["offensive_rating_diff", "defensive_rating_diff"],
        "mlb": ["starter_rating_diff", "hitting_rating_diff"],
        "nfl": ["epa_per_play_diff", "success_rate_diff", "qb_efficiency_diff"],
    }

    cols = checks.get(sport, [])
    if not cols:
        return

    validate_not_zero(df, cols, sport.upper())
