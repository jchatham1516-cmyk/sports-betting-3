"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _load_historical_csv(sport_name: str) -> pd.DataFrame | None:
    return _safe_read_csv(Path(f"sports_betting/data/historical/{sport_name}_historical.csv"))


def _derive_team_means_from_historical(
    historical_df: pd.DataFrame,
    metric_map: dict[str, list[str]],
) -> pd.DataFrame | None:
    team_rows: list[dict[str, object]] = []

    for _, row in historical_df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")

        if pd.notna(home_team):
            home_row: dict[str, object] = {"team": str(home_team).lower().strip()}
            for metric, candidates in metric_map.items():
                home_candidates = [f"{base}_home" for base in candidates]
                value = next((row.get(col) for col in home_candidates if col in row.index), None)
                home_row[metric] = pd.to_numeric(value, errors="coerce")
            team_rows.append(home_row)

        if pd.notna(away_team):
            away_row: dict[str, object] = {"team": str(away_team).lower().strip()}
            for metric, candidates in metric_map.items():
                away_candidates = [f"{base}_away" for base in candidates]
                value = next((row.get(col) for col in away_candidates if col in row.index), None)
                away_row[metric] = pd.to_numeric(value, errors="coerce")
            team_rows.append(away_row)

    team_df = pd.DataFrame(team_rows)
    if team_df.empty:
        return None
    return team_df.groupby("team", as_index=False).mean(numeric_only=True)


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

    metric_map = {
        "offensive_rating": ["offensive_rating", "off_rating"],
        "defensive_rating": ["defensive_rating", "def_rating"],
        "net_rating": ["net_rating"],
        "pace": ["pace"],
        "true_shooting": ["true_shooting"],
        "effective_fg": ["effective_fg", "efg"],
        "turnover_rate": ["turnover_rate"],
        "rebound_rate": ["rebound_rate"],
        "free_throw_rate": ["free_throw_rate"],
    }
    derived = _derive_team_means_from_historical(historical_df, metric_map)
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

    metric_map = {
        "goalie_save_strength": ["goalie_strength", "goalie_save_strength"],
        "special_teams_efficiency": ["special_teams_efficiency", "pp", "pk"],
        "xgf": ["xgf"],
        "xga": ["xga"],
        "shot_share": ["shot_share"],
    }
    derived = _derive_team_means_from_historical(historical_df, metric_map)
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

    metric_map = {
        "starter_rating": ["starter_rating"],
        "bullpen_rating": ["bullpen_rating"],
        "hitting_rating": ["hitting_rating"],
        "home_split": ["home_split"],
        "recent_form": ["recent_form"],
    }
    derived = _derive_team_means_from_historical(historical_df, metric_map)
    if derived is None:
        return None, "historical_empty"
    return derived, "historical_derived"


def _merge_home_away_team_stats(df: pd.DataFrame, team_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    stats = team_df.rename(columns={"team": "team_key"}).copy()

    def normalize_team(name: object) -> str:
        return str(name).lower().strip()

    stats["team_key"] = stats["team_key"].apply(normalize_team)

    if "home_team_norm" not in out.columns and "home_team" in out.columns:
        out["home_team_norm"] = out["home_team"].apply(normalize_team)
    if "away_team_norm" not in out.columns and "away_team" in out.columns:
        out["away_team_norm"] = out["away_team"].apply(normalize_team)

    target_columns = set(mapping.values()) | {dst.replace("_home", "_away") for dst in mapping.values()}
    cols_to_drop = [col for col in out.columns if col in target_columns or col.endswith("_x") or col.endswith("_y")]
    out = out.drop(columns=cols_to_drop, errors="ignore")

    home_stats = stats.rename(columns={src: dst for src, dst in mapping.items()})
    away_stats = stats.rename(columns={src: dst.replace("_home", "_away") for src, dst in mapping.items()})

    out = out.merge(home_stats, left_on="home_team_norm", right_on="team_key", how="left").drop(columns=["team_key"], errors="ignore")
    out = out.merge(away_stats, left_on="away_team_norm", right_on="team_key", how="left").drop(columns=["team_key"], errors="ignore")
    for base_col in mapping.values():
        away_col = base_col.replace("_home", "_away")
        if base_col in out.columns:
            out[base_col] = out[base_col]
        if away_col in out.columns:
            out[away_col] = out[away_col]
    if {"home_team", "offensive_rating_home", "offensive_rating_away"}.issubset(out.columns):
        print("[MERGE CHECK]")
        print(out[["home_team", "offensive_rating_home", "offensive_rating_away"]].head())
    return out


def enrich_daily_features_by_sport(df: pd.DataFrame, sport_name: str) -> pd.DataFrame:
    sport = str(sport_name).lower()

    if sport == "nba":
        from sports_betting.sports.nba.features import enrich_nba_live_features, build_nba_diff_features

        team_df, source = _resolve_nba_team_stats()
        print(f"[NBA ENRICHMENT SOURCE] {source}")
        if team_df is not None:
            mapping = {
                "offensive_rating": "offensive_rating_home",
                "defensive_rating": "defensive_rating_home",
                "net_rating": "net_rating_home",
                "pace": "pace_home",
                "true_shooting": "true_shooting_home",
                "effective_fg": "effective_fg_home",
                "turnover_rate": "turnover_rate_home",
                "rebound_rate": "rebound_rate_home",
                "free_throw_rate": "free_throw_rate_home",
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[NBA ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty"}:
                raise RuntimeError("[NBA DATA ERROR] No team stat source found after exhausting external and historical fallbacks.")
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
                "special_teams_efficiency": "special_teams_efficiency_home",
                "xgf": "xgf_home",
                "xga": "xga_home",
                "shot_share": "shot_share_home",
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[NHL ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty"}:
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
            }
            df = _merge_home_away_team_stats(df, team_df, mapping)
        else:
            print("[MLB ENRICHMENT WARNING] No team stat source found.")
            if source in {"missing", "historical_empty"}:
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
