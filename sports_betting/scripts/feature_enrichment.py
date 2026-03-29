"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sports_betting.sports.common.team_names import normalize_team_name


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _load_historical_csv(sport_name: str) -> pd.DataFrame | None:
    return _safe_read_csv(Path(f"sports_betting/data/historical/{sport_name}_historical.csv"))


def _normalize_team(name: object) -> str:
    return str(normalize_team_name(name))


def load_mlb_pitchers() -> dict[str, dict[str, float]]:
    candidates = [
        Path("data/inputs/mlb_pitchers.json"),
        Path("sports_betting/data/inputs/mlb_pitchers.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[MLB PITCHERS WARNING] Invalid JSON in {path}: {exc}")
            continue
        if isinstance(payload, dict):
            return {str(_normalize_team(team)): dict(values) for team, values in payload.items() if isinstance(values, dict)}
    return {}


def _normalize_pitcher_name(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def load_mlb_probable_pitchers() -> dict[str, dict[str, float | str]]:
    candidates = [
        Path("data/inputs/mlb_probable_pitchers.json"),
        Path("sports_betting/data/inputs/mlb_probable_pitchers.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[MLB PROBABLE WARNING] Invalid JSON in {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        normalized: dict[str, dict[str, float | str]] = {}
        for pitcher_name, values in payload.items():
            if not isinstance(values, dict):
                continue
            norm_name = _normalize_pitcher_name(pitcher_name)
            if not norm_name:
                continue
            normalized[norm_name] = dict(values)
        return normalized
    return {}


def _extract_pitcher_stats_from_row(
    row: pd.Series,
    side: str,
    probable_pitchers_by_name: dict[str, dict[str, float | str]],
    pitchers_by_team: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    probable_pitcher_columns = [
        f"{side}_probable_pitcher",
        f"{side}_pitcher",
        f"{side}_pitcher_name",
        f"probable_{side}_pitcher",
        f"probable_pitcher_{side}",
    ]
    for col in probable_pitcher_columns:
        if col not in row.index:
            continue
        pitcher_name = _normalize_pitcher_name(row.get(col))
        if not pitcher_name:
            continue
        probable_stats = probable_pitchers_by_name.get(pitcher_name, {})
        era_val = probable_stats.get("era")
        whip_val = probable_stats.get("whip")
        k_rate_val = probable_stats.get("k_rate")
        if era_val is not None or whip_val is not None or k_rate_val is not None:
            return {
                "pitcher_name": str(row.get(col)),
                "pitcher_era": float(era_val) if era_val is not None else float("nan"),
                "pitcher_whip": float(whip_val) if whip_val is not None else float("nan"),
                "pitcher_k_rate": float(k_rate_val) if k_rate_val is not None else float("nan"),
            }

    team_key = _normalize_team(row.get(f"{side}_team_norm"))
    team_defaults = pitchers_by_team.get(team_key, {})
    return {
        "pitcher_name": "",
        "pitcher_era": float(team_defaults.get("era")) if team_defaults.get("era") is not None else float("nan"),
        "pitcher_whip": float(team_defaults.get("whip")) if team_defaults.get("whip") is not None else float("nan"),
        "pitcher_k_rate": float(team_defaults.get("k_rate")) if team_defaults.get("k_rate") is not None else float("nan"),
    }





MLB_TEAM_ALIASES = {
    "sox": "sox",
    "red sox": "boston red sox",
    "white sox": "chicago white sox",
    "chi white sox": "chicago white sox",
    "chi cubs": "chicago cubs",
    "la dodgers": "los angeles dodgers",
    "la angels": "los angeles angels",
    "d backs": "arizona diamondbacks",
    "dbaks": "arizona diamondbacks",
}


def normalize_team(name: object) -> str:
    cleaned = str(name or "").lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned in MLB_TEAM_ALIASES:
        return MLB_TEAM_ALIASES[cleaned]
    if cleaned.endswith(" red sox"):
        return "boston red sox"
    if cleaned.endswith(" white sox"):
        return "chicago white sox"
    return cleaned


def _coerce_espn_game_rows(payload: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        text_blob = " ".join(str(v) for v in payload.values() if isinstance(v, (str, int, float))).lower()
        if "probable" in text_blob and "pitch" in text_blob:
            home_team = str(payload.get("homeTeam") or payload.get("home_team") or payload.get("home") or "").strip()
            away_team = str(payload.get("awayTeam") or payload.get("away_team") or payload.get("away") or "").strip()
            home_pitcher = str(
                payload.get("homeProbablePitcher")
                or payload.get("home_probable_pitcher")
                or payload.get("homePitcher")
                or payload.get("home_pitcher")
                or ""
            ).strip()
            away_pitcher = str(
                payload.get("awayProbablePitcher")
                or payload.get("away_probable_pitcher")
                or payload.get("awayPitcher")
                or payload.get("away_pitcher")
                or ""
            ).strip()
            if home_team and away_team and (home_pitcher or away_pitcher):
                rows.append(
                    {
                        "home_team_norm": normalize_team(home_team),
                        "away_team_norm": normalize_team(away_team),
                        "home_pitcher": home_pitcher,
                        "away_pitcher": away_pitcher,
                    }
                )
        for value in payload.values():
            rows.extend(_coerce_espn_game_rows(value))
    elif isinstance(payload, list):
        for item in payload:
            rows.extend(_coerce_espn_game_rows(item))
    return rows


def fetch_mlb_probable_pitchers_espn() -> pd.DataFrame:
    columns = ["home_team_norm", "away_team_norm", "home_pitcher", "away_pitcher"]
    print("\n[ESPN SCRAPER STARTED]")
    try:
        response = requests.get(
            "https://www.espn.com/mlb/scoreboard",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[MLB ESPN WARNING] ESPN request failed: {exc}")
        return pd.DataFrame([{"home_team_norm": "__unknown_home__", "away_team_norm": "__unknown_away__", "home_pitcher": None, "away_pitcher": None}], columns=columns)

    print("[HTML LENGTH]", len(response.text))
    if len(response.text) < 10000:
        print("WARNING: ESPN page likely blocked or incomplete")

    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, str]] = []

    game_cards = soup.find_all("section")
    for card in game_cards:
        team_nodes = card.find_all(["span", "h2"])
        team_candidates: list[str] = []
        for node in team_nodes:
            text_value = node.get_text(" ", strip=True)
            if not text_value:
                continue
            lowered = text_value.lower()
            if any(token in lowered for token in ("mlb", "probable", "pitcher", "final", "top", "bottom", "at ", "vs")):
                continue
            if len(text_value) < 3 or len(text_value) > 35:
                continue
            if re.search(r"\d", text_value):
                continue
            team_candidates.append(text_value)
        dedup_candidates: list[str] = []
        for candidate in team_candidates:
            if candidate.lower() not in {v.lower() for v in dedup_candidates}:
                dedup_candidates.append(candidate)

        if len(dedup_candidates) >= 2:
            away_team = dedup_candidates[0]
            home_team = dedup_candidates[1]
            section_text = card.get_text(" ", strip=True)
            probable_names: list[str] = []
            if "probable" in section_text.lower():
                for match in re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z\-\']+)+)", section_text):
                    snippet = str(match).strip()
                    if not snippet:
                        continue
                    if snippet.lower() in {away_team.lower(), home_team.lower()}:
                        continue
                    if "probable" in snippet.lower():
                        continue
                    probable_names.append(snippet)

            away_pitcher = probable_names[0] if len(probable_names) >= 1 else None
            home_pitcher = probable_names[1] if len(probable_names) >= 2 else None
            rows.append(
                {
                    "home_team_norm": normalize_team(home_team),
                    "away_team_norm": normalize_team(away_team),
                    "home_pitcher": home_pitcher,
                    "away_pitcher": away_pitcher,
                }
            )

    if not rows:
        for script in soup.find_all("script"):
            payload_text = script.string or script.get_text("", strip=True)
            if not payload_text or "probable" not in payload_text.lower() or "pitch" not in payload_text.lower():
                continue
            for candidate in re.finditer(r"\{.*\}", payload_text):
                fragment = candidate.group(0)
                try:
                    decoded = json.loads(fragment)
                except json.JSONDecodeError:
                    continue
                rows.extend(_coerce_espn_game_rows(decoded))
                if rows:
                    break
            if rows:
                break

    if not rows:
        rows.append(
            {
                "home_team_norm": "__unknown_home__",
                "away_team_norm": "__unknown_away__",
                "home_pitcher": None,
                "away_pitcher": None,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    out = out.drop_duplicates(subset=["home_team_norm", "away_team_norm"], keep="last")
    print("\n[ESPN PITCHER DEBUG]")
    print(out.head(20))
    print("TOTAL GAMES SCRAPED:", len(out))
    return out


def _normalize_mlb_team_key(name: object) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", str(name or "").strip().lower())
    return " ".join(cleaned.split())


def _extract_probable_pitcher_name(payload: object) -> str:
    if isinstance(payload, dict):
        candidate = payload.get("full_name") or payload.get("name")
        return str(candidate or "").strip()
    return ""


def fetch_mlb_probable_pitchers(target_date: date | datetime | str | None = None) -> pd.DataFrame:
    api_key = os.getenv("SPORTSRADAR_API_KEY") or os.getenv("SPORTRADAR_API_KEY")
    columns = [
        "home_team_norm",
        "away_team_norm",
        "home_probable_pitcher",
        "away_probable_pitcher",
    ]
    if not api_key:
        print("[MLB PROBABLE WARNING] Missing SportsRadar API key (SPORTSRADAR_API_KEY/SPORTRADAR_API_KEY).")
        return pd.DataFrame(columns=columns)

    if target_date is None:
        game_date = datetime.now(UTC).date()
    elif isinstance(target_date, datetime):
        game_date = target_date.date()
    elif isinstance(target_date, date):
        game_date = target_date
    else:
        game_date = datetime.fromisoformat(str(target_date)).date()

    rows: list[dict[str, str]] = []
    date_token = game_date.strftime("%Y/%m/%d")
    for access_level in ("trial", "production"):
        schedule_url = f"https://api.sportradar.com/mlb/{access_level}/v7/en/games/{date_token}/schedule.json"
        try:
            response = requests.get(schedule_url, params={"api_key": api_key}, timeout=15)
            if response.status_code >= 400:
                continue
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[MLB PROBABLE WARNING] SportsRadar request failed: {exc}")
            continue

        for game in payload.get("games", []):
            home = game.get("home", {}) if isinstance(game, dict) else {}
            away = game.get("away", {}) if isinstance(game, dict) else {}
            home_name = home.get("name", "")
            away_name = away.get("name", "")
            home_probable = (
                _extract_probable_pitcher_name(game.get("home_probable_pitcher"))
                or _extract_probable_pitcher_name(home.get("probable_pitcher"))
            )
            away_probable = (
                _extract_probable_pitcher_name(game.get("away_probable_pitcher"))
                or _extract_probable_pitcher_name(away.get("probable_pitcher"))
            )
            if not home_name or not away_name:
                continue
            rows.append(
                {
                    "home_team_norm": _normalize_mlb_team_key(home_name),
                    "away_team_norm": _normalize_mlb_team_key(away_name),
                    "home_probable_pitcher": home_probable,
                    "away_probable_pitcher": away_probable,
                }
            )

        if rows:
            break

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["home_team_norm", "away_team_norm"], keep="last")

def load_nhl_goalies() -> dict[str, dict[str, float]]:
    candidates = [
        Path("data/inputs/nhl_goalies.json"),
        Path("sports_betting/data/inputs/nhl_goalies.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[NHL GOALIES WARNING] Invalid JSON in {path}: {exc}")
            continue
        if isinstance(payload, dict):
            return {str(_normalize_team(team)): dict(values) for team, values in payload.items() if isinstance(values, dict)}
    return {}


def build_nba_team_stats(historical: pd.DataFrame) -> pd.DataFrame:
    frame = historical.copy()

    def norm(x: object) -> str:
        return str(normalize_team_name(x))

    frame["home_team_norm"] = frame["home_team"].apply(norm)
    frame["away_team_norm"] = frame["away_team"].apply(norm)

    # Ensure we always derive score-based stats even when advanced columns are absent.
    score_aliases = {
        "home": ["home_score", "home_points", "pts_home", "home_pts"],
        "away": ["away_score", "away_points", "pts_away", "away_pts"],
    }

    home_col = next((col for col in score_aliases["home"] if col in frame.columns), None)
    away_col = next((col for col in score_aliases["away"] if col in frame.columns), None)

    home_scores = frame[home_col] if home_col else pd.Series(0.0, index=frame.index)
    away_scores = frame[away_col] if away_col else pd.Series(0.0, index=frame.index)
    frame["home_score"] = pd.to_numeric(home_scores, errors="coerce").fillna(0.0)
    frame["away_score"] = pd.to_numeric(away_scores, errors="coerce").fillna(0.0)
    frame["point_diff"] = frame["home_score"] - frame["away_score"]

    home = (
        frame.groupby("home_team_norm")
        .agg(
            {
                "home_score": "mean",
                "away_score": "mean",
                "point_diff": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "home_team_norm": "team_norm",
                "home_score": "points_for",
                "away_score": "points_against",
            }
        )
    )

    away = (
        frame.groupby("away_team_norm")
        .agg(
            {
                "away_score": "mean",
                "home_score": "mean",
                "point_diff": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "away_team_norm": "team_norm",
                "away_score": "points_for",
                "home_score": "points_against",
            }
        )
    )

    team_stats = pd.concat([home, away], ignore_index=True)
    team_stats = team_stats.groupby("team_norm", as_index=False).mean(numeric_only=True)
    return team_stats


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

        fallback_point_diff = None
        if {"points_for_home", "points_against_home", "points_for_away", "points_against_away"}.issubset(df.columns):
            fallback_point_diff = (
                pd.to_numeric(df["points_for_home"], errors="coerce").fillna(0.0)
                - pd.to_numeric(df["points_against_home"], errors="coerce").fillna(0.0)
                - pd.to_numeric(df["points_for_away"], errors="coerce").fillna(0.0)
                + pd.to_numeric(df["points_against_away"], errors="coerce").fillna(0.0)
            )

        team_df, source = _resolve_nba_team_stats()
        print(f"[NBA ENRICHMENT SOURCE] {source}")
        if source == "historical_derived":
            print("[NBA INFO] Using basic stat fallback (score-based features)")
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
        if "point_diff_diff" in df.columns:
            df["offensive_rating_diff"] = df["point_diff_diff"]
            df["defensive_rating_diff"] = -df["point_diff_diff"]
        df = enrich_nba_live_features(df, nba_team_stats=None)
        df = build_nba_diff_features(df)
        if fallback_point_diff is not None:
            df["point_diff_diff"] = fallback_point_diff
        elif {"points_for_home", "points_against_home", "points_for_away", "points_against_away"}.issubset(df.columns):
            df["point_diff_diff"] = (df["points_for_home"] - df["points_against_home"]) - (
                df["points_for_away"] - df["points_against_away"]
            )
        if "point_diff_diff" in df.columns:
            off_signal = pd.to_numeric(df.get("offensive_rating_diff", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
            if off_signal.abs().sum() == 0:
                df["offensive_rating_diff"] = pd.to_numeric(df["point_diff_diff"], errors="coerce").fillna(0.0)
                df["defensive_rating_diff"] = -pd.to_numeric(df["point_diff_diff"], errors="coerce").fillna(0.0)
        print("Non-zero offensive_diff:", (pd.to_numeric(df.get("offensive_rating_diff"), errors="coerce").fillna(0.0) != 0).sum())
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
        goalies = load_nhl_goalies()
        df["goalie_save_home"] = df["home_team_norm"].map(lambda t: goalies.get(str(t), {}).get("save_pct", 0.0))
        df["goalie_save_away"] = df["away_team_norm"].map(lambda t: goalies.get(str(t), {}).get("save_pct", 0.0))
        df["goalie_save_home"] = pd.to_numeric(df["goalie_save_home"], errors="coerce").fillna(0.905)
        df["goalie_save_away"] = pd.to_numeric(df["goalie_save_away"], errors="coerce").fillna(0.905)
        df["goalie_diff"] = df["goalie_save_home"] - df["goalie_save_away"]
        print("Non-zero goalie_diff:", (df["goalie_diff"] != 0).sum())
        return df

    if sport == "mlb":
        print("\n🚨 MLB PIPELINE STARTED 🚨")
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
        pitchers = load_mlb_pitchers()
        probable_pitchers = load_mlb_probable_pitchers()

        if "home_team_norm" not in df.columns:
            df["home_team_norm"] = df["home_team"].map(_normalize_team)
        if "away_team_norm" not in df.columns:
            df["away_team_norm"] = df["away_team"].map(_normalize_team)

        print("\n🚨 BEFORE SCRAPER 🚨")
        from sports_betting.data_collection.espn_mlb_pitchers import fetch_mlb_probable_pitchers_espn

        pitcher_df = fetch_mlb_probable_pitchers_espn()

        print("\n🚨 AFTER SCRAPER 🚨")

        if pitcher_df is None:
            print("❌ pitcher_df is NONE")
            pitcher_df = pd.DataFrame(columns=["home_team_norm", "away_team_norm", "home_pitcher", "away_pitcher"])
        elif pitcher_df.empty:
            print("❌ pitcher_df is EMPTY")
        else:
            print("✅ pitcher_df WORKING")
            print(pitcher_df.head())

        if not pitcher_df.empty:
            pitcher_df["home_team_norm"] = pitcher_df["home_team_norm"].map(_normalize_team)
            pitcher_df["away_team_norm"] = pitcher_df["away_team_norm"].map(_normalize_team)
        try:
            df = df.merge(
                pitcher_df,
                on=["home_team_norm", "away_team_norm"],
                how="left",
            )
            print("✅ MERGE ATTEMPTED")
        except Exception as e:
            print("❌ MERGE FAILED:", str(e))

        if "home_probable_pitcher" not in df.columns:
            df["home_probable_pitcher"] = df.get("home_probable_pitcher", "")
        if "away_probable_pitcher" not in df.columns:
            df["away_probable_pitcher"] = df.get("away_probable_pitcher", "")
        if "home_pitcher" not in df.columns:
            df["home_pitcher"] = None
        if "away_pitcher" not in df.columns:
            df["away_pitcher"] = None
        df["home_probable_pitcher"] = df["home_probable_pitcher"].astype(str).where(
            df["home_probable_pitcher"].astype(str).str.strip().ne(""),
            df["home_pitcher"].fillna("").astype(str),
        )
        df["away_probable_pitcher"] = df["away_probable_pitcher"].astype(str).where(
            df["away_probable_pitcher"].astype(str).str.strip().ne(""),
            df["away_pitcher"].fillna("").astype(str),
        )

        print("\n[ESPN PITCHER DEBUG]")
        print(pitcher_df.head(10) if 'pitcher_df' in locals() else pd.DataFrame())
        print("TOTAL GAMES SCRAPED:", len(pitcher_df))

        print("\n[POST-MERGE CHECK]")
        print(df[["home_team", "away_team", "home_pitcher", "away_pitcher"]].head(10))
        print("\n🚨 MLB FINAL CHECK 🚨")
        print(df[["home_team", "away_team"]].head())

        for side in ("home", "away"):
            probable_col = f"{side}_probable_pitcher"
            target_col = f"probable_{side}_pitcher"
            if target_col not in df.columns:
                df[target_col] = ""
            if probable_col in df.columns:
                df[target_col] = df[target_col].astype(str).where(df[target_col].astype(str).str.strip().ne(""), df[probable_col].fillna("").astype(str))

        home_pitcher_stats = df.apply(
            lambda row: _extract_pitcher_stats_from_row(
                row,
                side="home",
                probable_pitchers_by_name=probable_pitchers,
                pitchers_by_team=pitchers,
            ),
            axis=1,
        )
        away_pitcher_stats = df.apply(
            lambda row: _extract_pitcher_stats_from_row(
                row,
                side="away",
                probable_pitchers_by_name=probable_pitchers,
                pitchers_by_team=pitchers,
            ),
            axis=1,
        )

        df["pitcher_name_home"] = home_pitcher_stats.map(lambda v: v.get("pitcher_name", "")).astype(str)
        df["pitcher_name_away"] = away_pitcher_stats.map(lambda v: v.get("pitcher_name", "")).astype(str)
        df["pitcher_era_home"] = pd.to_numeric(home_pitcher_stats.map(lambda v: v.get("pitcher_era")), errors="coerce")
        df["pitcher_era_away"] = pd.to_numeric(away_pitcher_stats.map(lambda v: v.get("pitcher_era")), errors="coerce")
        df["pitcher_whip_home"] = pd.to_numeric(home_pitcher_stats.map(lambda v: v.get("pitcher_whip")), errors="coerce")
        df["pitcher_whip_away"] = pd.to_numeric(away_pitcher_stats.map(lambda v: v.get("pitcher_whip")), errors="coerce")
        df["pitcher_k_rate_home"] = pd.to_numeric(home_pitcher_stats.map(lambda v: v.get("pitcher_k_rate")), errors="coerce")
        df["pitcher_k_rate_away"] = pd.to_numeric(away_pitcher_stats.map(lambda v: v.get("pitcher_k_rate")), errors="coerce")

        missing_pitchers = df[
            df["pitcher_name_home"].eq("")
            | df["pitcher_name_away"].eq("")
            | df["pitcher_era_home"].isna()
            | df["pitcher_era_away"].isna()
        ]
        if not missing_pitchers.empty:
            print("[MLB PITCHER MISSING]")
            print(
                missing_pitchers[
                    [
                        "home_team",
                        "away_team",
                        "pitcher_name_home",
                        "pitcher_name_away",
                        "pitcher_era_home",
                        "pitcher_era_away",
                    ]
                ].to_string(index=False)
            )

        df["pitcher_era_home"] = df["home_pitcher"].apply(lambda x: 4.2 if pd.isna(x) else 3.5)
        df["pitcher_era_away"] = df["away_pitcher"].apply(lambda x: 4.2 if pd.isna(x) else 3.5)
        df["pitcher_era_home"] = df["pitcher_era_home"].fillna(pd.to_numeric(df.get("starter_rating_home"), errors="coerce").rsub(125).div(25))
        df["pitcher_era_away"] = df["pitcher_era_away"].fillna(pd.to_numeric(df.get("starter_rating_away"), errors="coerce").rsub(125).div(25))
        df["pitcher_whip_home"] = df["pitcher_whip_home"].fillna((df["pitcher_era_home"] / 4.0).clip(lower=0.9, upper=1.8))
        df["pitcher_whip_away"] = df["pitcher_whip_away"].fillna((df["pitcher_era_away"] / 4.0).clip(lower=0.9, upper=1.8))
        df["pitcher_k_rate_home"] = df["pitcher_k_rate_home"].fillna((0.30 - (df["pitcher_era_home"] - 3.0) * 0.025).clip(lower=0.12, upper=0.35))
        df["pitcher_k_rate_away"] = df["pitcher_k_rate_away"].fillna((0.30 - (df["pitcher_era_away"] - 3.0) * 0.025).clip(lower=0.12, upper=0.35))
        df["pitcher_diff"] = df["pitcher_era_away"] - df["pitcher_era_home"]
        df["pitcher_whip_diff"] = df["pitcher_whip_away"] - df["pitcher_whip_home"]
        df["pitcher_k_rate_diff"] = df["pitcher_k_rate_home"] - df["pitcher_k_rate_away"]
        print("\n[PITCHER DEBUG MLB]")
        print(df[["home_team", "away_team", "pitcher_era_home", "pitcher_era_away", "pitcher_diff"]].head(10))

        PITCHER_WEIGHT = 0.015
        if "edge" in df.columns:
            if "adjusted_edge" in df.columns:
                df["adjusted_edge"] = pd.to_numeric(df["adjusted_edge"], errors="coerce")
                df["adjusted_edge"] = df["adjusted_edge"].where(df["adjusted_edge"].notna(), pd.to_numeric(df["edge"], errors="coerce") + (df["pitcher_diff"] * PITCHER_WEIGHT))
            else:
                df["adjusted_edge"] = pd.to_numeric(df["edge"], errors="coerce") + (df["pitcher_diff"] * PITCHER_WEIGHT)
        df = enrich_mlb_live_features(df)
        df["starter_rating_home"] = pd.to_numeric(df["starter_rating_home"], errors="coerce").replace(0, 50).fillna(50)
        df["starter_rating_away"] = pd.to_numeric(df["starter_rating_away"], errors="coerce").replace(0, 50).fillna(50)
        df["hitting_rating_home"] = pd.to_numeric(df["hitting_rating_home"], errors="coerce").replace(0, 100).fillna(100)
        df["hitting_rating_away"] = pd.to_numeric(df["hitting_rating_away"], errors="coerce").replace(0, 100).fillna(100)
        df["starter_rating_diff"] = df["starter_rating_home"] - df["starter_rating_away"]
        df["hitting_rating_diff"] = df["hitting_rating_home"] - df["hitting_rating_away"]
        print("Non-zero starter_diff:", (df["starter_rating_diff"] != 0).sum())
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

    if sport == "nba" and "offensive_rating_diff" in df.columns:
        print("[FEATURE CHECK]")
        print(df[["offensive_rating_diff"]].describe())

    validate_not_zero(df, cols, sport.upper())
