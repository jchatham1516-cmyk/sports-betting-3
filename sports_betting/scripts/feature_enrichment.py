"""Sport-specific live feature enrichment and validation utilities."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sports_betting.data_collection.pitcher_stats import build_pitcher_era_records
from sports_betting.sports.common.name_normalization import normalize_person_name
from sports_betting.sports.common.team_names import normalize_team_name

DEFAULT_MLB_ERA = 4.20
MLB_REAL_ERA_NORMAL_THRESHOLD = 50.0
MLB_REAL_ERA_SEVERE_THRESHOLD = 25.0


def _is_real_mlb_era_value(value: object) -> bool:
    era = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(era) and era > 0 and era <= 15 and not np.isclose(float(era), DEFAULT_MLB_ERA))


def _mlb_quality_from_real_coverage(real_coverage_pct: float) -> str:
    if real_coverage_pct < MLB_REAL_ERA_SEVERE_THRESHOLD:
        return "severe"
    if real_coverage_pct < MLB_REAL_ERA_NORMAL_THRESHOLD:
        return "degraded"
    return "normal"


def safe_date(value):
    """Return a date for datetime-like values without calling .date() on date objects."""
    if hasattr(value, "date") and not isinstance(value, date):
        return value.date()
    return value


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _load_historical_csv(sport_name: str) -> pd.DataFrame | None:
    return _safe_read_csv(Path(f"sports_betting/data/historical/{sport_name}_historical.csv"))


def _normalize_team(name: object) -> str:
    return str(normalize_team_name(name))


NHL_GOALIE_TEAM_ALIASES = {
    "anaheim": "anaheim ducks",
    "arizona coyotes": "utah mammoth",
    "boston": "boston bruins",
    "buffalo": "buffalo sabres",
    "calgary": "calgary flames",
    "carolina": "carolina hurricanes",
    "chicago": "chicago blackhawks",
    "colorado": "colorado avalanche",
    "columbus": "columbus blue jackets",
    "dallas": "dallas stars",
    "detroit": "detroit red wings",
    "edmonton": "edmonton oilers",
    "florida": "florida panthers",
    "los angeles": "los angeles kings",
    "la kings": "los angeles kings",
    "minnesota": "minnesota wild",
    "montreal": "montreal canadiens",
    "nashville": "nashville predators",
    "new jersey": "new jersey devils",
    "new york islanders": "new york islanders",
    "new york rangers": "new york rangers",
    "ny islanders": "new york islanders",
    "ny rangers": "new york rangers",
    "ottawa": "ottawa senators",
    "philadelphia": "philadelphia flyers",
    "pittsburgh": "pittsburgh penguins",
    "san jose": "san jose sharks",
    "seattle": "seattle kraken",
    "st louis": "st louis blues",
    "st louis blues": "st louis blues",
    "tampa bay": "tampa bay lightning",
    "toronto": "toronto maple leafs",
    "utah hockey club": "utah mammoth",
    "utah mammoth": "utah mammoth",
    "vancouver": "vancouver canucks",
    "vegas": "vegas golden knights",
    "vegas golden knights": "vegas golden knights",
    "washington": "washington capitals",
    "winnipeg": "winnipeg jets",
}

NHL_TEAM_ABBREVIATIONS = {
    "ANA": "anaheim ducks",
    "BOS": "boston bruins",
    "BUF": "buffalo sabres",
    "CAR": "carolina hurricanes",
    "CBJ": "columbus blue jackets",
    "CGY": "calgary flames",
    "CHI": "chicago blackhawks",
    "COL": "colorado avalanche",
    "DAL": "dallas stars",
    "DET": "detroit red wings",
    "EDM": "edmonton oilers",
    "FLA": "florida panthers",
    "LAK": "los angeles kings",
    "MIN": "minnesota wild",
    "MTL": "montreal canadiens",
    "NJD": "new jersey devils",
    "NSH": "nashville predators",
    "NYI": "new york islanders",
    "NYR": "new york rangers",
    "OTT": "ottawa senators",
    "PHI": "philadelphia flyers",
    "PIT": "pittsburgh penguins",
    "SEA": "seattle kraken",
    "SJS": "san jose sharks",
    "STL": "st louis blues",
    "TBL": "tampa bay lightning",
    "TOR": "toronto maple leafs",
    "UTA": "utah mammoth",
    "VAN": "vancouver canucks",
    "VGK": "vegas golden knights",
    "WPG": "winnipeg jets",
    "WSH": "washington capitals",
}

NHL_NEUTRAL_GOALIE_SAVE_PCT = 0.905
NHL_GOALIE_API_DELAY_SECONDS = 0.25
NHL_GOALIE_API_STATS = {
    "requests_attempted": 0,
    "successful": 0,
    "rate_limited": 0,
    "cache_hits": 0,
}
NHL_GOALIE_TEAM_STATS_CACHE: dict[str, dict[str, object]] = {}
NHL_GOALIE_RATE_LIMITED = False


def _reset_nhl_goalie_api_status() -> None:
    global NHL_GOALIE_RATE_LIMITED
    NHL_GOALIE_RATE_LIMITED = False
    for key in NHL_GOALIE_API_STATS:
        NHL_GOALIE_API_STATS[key] = 0


def _mark_nhl_goalie_rate_limited() -> None:
    global NHL_GOALIE_RATE_LIMITED
    NHL_GOALIE_RATE_LIMITED = True
    NHL_GOALIE_API_STATS["rate_limited"] += 1


def _print_nhl_goalie_api_status(coverage_pct: float) -> None:
    print("[NHL GOALIE API STATUS]")
    print(f"requests_attempted: {NHL_GOALIE_API_STATS['requests_attempted']}")
    print(f"successful: {NHL_GOALIE_API_STATS['successful']}")
    print(f"rate_limited: {NHL_GOALIE_API_STATS['rate_limited']}")
    print(f"cache_hits: {NHL_GOALIE_API_STATS['cache_hits']}")
    print(f"coverage_pct: {coverage_pct:.1f}")


def _nhl_api_get_json(url: str) -> dict[str, object] | None:
    if NHL_GOALIE_RATE_LIMITED:
        return None
    NHL_GOALIE_API_STATS["requests_attempted"] += 1
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException as exc:
        print(f"[NHL GOALIES WARNING] NHL request failed for {url}: {exc}")
        return None
    if response.status_code == 429:
        _mark_nhl_goalie_rate_limited()
        print("[NHL GOALIES WARNING] NHL API returned 429; disabling live goalie fetches for this run")
        return None
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[NHL GOALIES WARNING] NHL request failed for {url}: {exc}")
        return None
    NHL_GOALIE_API_STATS["successful"] += 1
    time.sleep(NHL_GOALIE_API_DELAY_SECONDS)
    return payload if isinstance(payload, dict) else None


def _normalize_goalie_team_key(name: object) -> str:
    raw = str(name or "")
    cleaned = str(normalize_team_name(raw)).lower().strip()
    cleaned = cleaned.replace(".", "").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned == "utah hockey club" and "mammoth" in raw.lower():
        cleaned = "utah mammoth"
    return NHL_GOALIE_TEAM_ALIASES.get(cleaned, cleaned)


def _extract_nhl_display_value(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("default", "en", "fr", "name", "displayName"):
            value = payload.get(key)
            if value:
                return str(value)
    if payload is None:
        return ""
    return str(payload)


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
            return {str(_normalize_team_key(team)): dict(values) for team, values in payload.items() if isinstance(values, dict)}
    return {}


def _normalize_pitcher_name(name: object) -> str:
    return normalize_person_name(name)


def _standardize_mlb_pitcher_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map known pitcher-name inputs into canonical MLB pitcher columns."""
    out = df.copy()
    candidates = {
        "home": ["home_pitcher", "pitcher_name_home", "home_probable_pitcher", "probable_home_pitcher", "probable_pitcher_home", "starting_pitcher_home", "home_pitcher_name"],
        "away": ["away_pitcher", "pitcher_name_away", "away_probable_pitcher", "probable_away_pitcher", "probable_pitcher_away", "starting_pitcher_away", "away_pitcher_name"],
    }
    for side, cols in candidates.items():
        canonical = f"{side}_pitcher"
        name_col = f"pitcher_name_{side}"
        if canonical not in out.columns:
            out[canonical] = ""
        if name_col not in out.columns:
            out[name_col] = ""
        for col in cols:
            if col not in out.columns:
                continue
            values = out[col].fillna("").astype(str)
            has_value = values.str.strip().ne("")
            out[canonical] = out[canonical].fillna("").astype(str).where(out[canonical].fillna("").astype(str).str.strip().ne(""), values.where(has_value, ""))
            out[name_col] = out[name_col].fillna("").astype(str).where(out[name_col].fillna("").astype(str).str.strip().ne(""), values.where(has_value, ""))
    return out


def _convert_pitcher_era_to_rating(era: pd.Series) -> pd.Series:
    return (100 - (pd.to_numeric(era, errors="coerce") * 10)).clip(lower=40, upper=90)


def _populate_mlb_starter_ratings_from_era(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ("home", "away"):
        rating_col = f"starter_rating_{side}"
        era_col = f"pitcher_era_{side}"
        if rating_col not in out.columns:
            out[rating_col] = np.nan
        if era_col not in out.columns:
            continue
        rating = pd.to_numeric(out[rating_col], errors="coerce")
        era = pd.to_numeric(out[era_col], errors="coerce")
        missing_rating = rating.isna() | rating.eq(0)
        out[rating_col] = rating.where(~(missing_rating & era.notna()), _convert_pitcher_era_to_rating(era))
    out["starter_rating_diff"] = pd.to_numeric(out.get("starter_rating_home"), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("starter_rating_away"), errors="coerce").fillna(0.0)
    return out


def load_mlb_probable_pitchers() -> dict[str, dict[str, float | str]]:
    normalized: dict[str, dict[str, float | str]] = {}
    for norm_name, record in build_pitcher_era_records().items():
        normalized[norm_name] = {
            "era": float(record["era"]),
            "source": str(record.get("source", "unknown")),
        }

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
        for pitcher_name, values in payload.items():
            if not isinstance(values, dict):
                continue
            norm_name = _normalize_pitcher_name(pitcher_name)
            if not norm_name:
                continue
            normalized.setdefault(norm_name, dict(values))
        break
    return normalized


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
    first_pitcher_name = ""
    for col in probable_pitcher_columns:
        if col not in row.index:
            continue
        pitcher_name = _normalize_pitcher_name(row.get(col))
        if not pitcher_name:
            continue
        if not first_pitcher_name:
            first_pitcher_name = str(row.get(col) or "").strip()
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
                "pitcher_era_is_real": bool(_normalize_pitcher_name(row.get(col)) and _is_real_mlb_era_value(era_val)),
                "pitcher_era_source": str(probable_stats.get("source", "probable_pitcher_stats")),
            }

    team_key = _normalize_team(row.get(f"{side}_team_norm"))
    team_defaults = pitchers_by_team.get(team_key, {})
    return {
        "pitcher_name": first_pitcher_name,
        "pitcher_era": float(team_defaults.get("era")) if team_defaults.get("era") is not None else float("nan"),
        "pitcher_whip": float(team_defaults.get("whip")) if team_defaults.get("whip") is not None else float("nan"),
        "pitcher_k_rate": float(team_defaults.get("k_rate")) if team_defaults.get("k_rate") is not None else float("nan"),
        "pitcher_era_is_real": False,
        "pitcher_era_source": "team_default",
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

TEAM_MAP = {
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "athletics": "athletics",
    "oakland a's": "athletics",
    "oakland a s": "athletics",
    "oakland as": "athletics",
    "st. louis cardinals": "st louis cardinals",
    "st louis cardinals": "st louis cardinals",
    "la angels": "los angeles angels",
    "los angeles angels": "los angeles angels",
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




def clean_team_name(name: object) -> str:
    return (
        str(name or "").lower()
        .strip()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )


def _normalize_team_key(name: object) -> str:
    cleaned = clean_team_name(name)
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return TEAM_MAP.get(cleaned, cleaned)

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


def get_mlb_starting_pitchers() -> pd.DataFrame:
    url = "https://www.espn.com/mlb/scoreboard"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[MLB ESPN WARNING] Could not fetch starting pitchers: {exc}")
        return pd.DataFrame(columns=["away_team", "home_team", "away_pitcher", "home_pitcher"])
    soup = BeautifulSoup(response.text, "html.parser")
    games: list[dict[str, str]] = []
    for game in soup.find_all("section", {"class": "Scoreboard"}):
        teams = game.find_all("span", {"class": "sb-team-short"})
        pitchers = game.find_all("div", {"class": "Pitcher"})
        if len(teams) == 2 and len(pitchers) >= 2:
            games.append(
                {
                    "away_team": teams[0].text.lower(),
                    "home_team": teams[1].text.lower(),
                    "away_pitcher": pitchers[0].text.lower(),
                    "home_pitcher": pitchers[1].text.lower(),
                }
            )
    return pd.DataFrame(games)


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
    elif isinstance(target_date, (datetime, date)):
        game_date = safe_date(target_date)
    else:
        game_date = safe_date(datetime.fromisoformat(str(target_date)))

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

def _extract_nhl_team_display_name(team_payload: dict) -> str:
    for key in ("default", "en", "fr"):
        value = team_payload.get(key) if isinstance(team_payload, dict) else None
        if value:
            return str(value)
    return ""


def _nhl_schedule_dates() -> list[str]:
    today = date.today()
    return [
        safe_date(today).isoformat(),
        safe_date(today + pd.Timedelta(days=1)).isoformat(),
        safe_date(today - pd.Timedelta(days=1)).isoformat(),
    ]


def _extract_goalie_name_from_payload(payload: object, side: str) -> str:
    """Best-effort extraction for NHL API goalie fields across schema variants."""
    wanted_side = str(side or "").lower()
    candidates: list[str] = []

    def walk(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            lower_keys = {str(k).lower(): k for k in value.keys()}
            path_text = " ".join(path).lower()
            if "goalie" in path_text or "startinggoalie" in path_text or "probablegoalie" in path_text:
                display = _extract_nhl_display_value(value.get("name") or value.get("fullName") or value.get("displayName"))
                if display:
                    candidates.append(display)
            for goalie_key in ("startingGoalie", "probableGoalie", "confirmedGoalie", "goalie"):
                actual = lower_keys.get(goalie_key.lower())
                if actual is not None:
                    display = _extract_nhl_display_value(value.get(actual))
                    if display and not isinstance(value.get(actual), (dict, list)):
                        candidates.append(display)
                    walk(value.get(actual), path + (goalie_key,))
            for key, child in value.items():
                walk(child, path + (str(key),))
        elif isinstance(value, list):
            for child in value:
                walk(child, path)

    team_payload = payload.get(f"{wanted_side}Team") if isinstance(payload, dict) else None
    if team_payload:
        walk(team_payload, (wanted_side, "team"))
    walk(payload, ())
    return next((name for name in candidates if name and name.lower() not in {"home", "away"}), "")


def _fetch_nhl_probable_goalies_from_api() -> dict[str, dict[str, object]]:
    """Fetch confirmed/probable goalie names when the public NHL payload exposes them.

    The NHL API does not always publish starters. This best-effort layer is used
    before team season save-percentage fallback and safely returns partial data.
    Rate limits stop additional live requests for the remainder of the run.
    """
    goalies: dict[str, dict[str, object]] = {}
    game_ids: set[int] = set()
    for game_date in _nhl_schedule_dates():
        payload = _nhl_api_get_json(f"https://api-web.nhle.com/v1/schedule/{game_date}")
        if payload is None:
            if NHL_GOALIE_RATE_LIMITED:
                return goalies
            continue
        for week in payload.get("gameWeek", []):
            for game in week.get("games", []):
                game_id = game.get("id")
                if game_id:
                    game_ids.add(int(game_id))

    for game_id in sorted(game_ids):
        payload = _nhl_api_get_json(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing")
        if payload is None:
            if NHL_GOALIE_RATE_LIMITED:
                return goalies
            continue
        for side in ("home", "away"):
            team = payload.get(f"{side}Team", {}) if isinstance(payload, dict) else {}
            team_key = _normalize_goalie_team_key(
                _extract_nhl_display_value(team.get("name"))
                or _extract_nhl_display_value(team.get("placeName"))
                or team.get("abbrev", "")
            )
            if not team_key or team_key in NHL_TEAM_ABBREVIATIONS:
                team_key = _normalize_goalie_team_key(NHL_TEAM_ABBREVIATIONS.get(str(team.get("abbrev", "")).upper(), team_key))
            goalie_name = _extract_goalie_name_from_payload(payload, side)
            if goalie_name:
                goalies[team_key] = {"goalie": goalie_name, "source": "nhl_api_probable_goalie"}
    return goalies


def _fetch_nhl_goalies_from_api() -> dict[str, dict[str, float]]:
    if NHL_GOALIE_RATE_LIMITED:
        return {}
    _reset_nhl_goalie_api_status()
    probable_goalies = _fetch_nhl_probable_goalies_from_api()
    if NHL_GOALIE_RATE_LIMITED:
        return probable_goalies

    standings_payload = _nhl_api_get_json("https://api-web.nhle.com/v1/standings/now")
    if standings_payload is None:
        return probable_goalies

    teams_by_abbrev: dict[str, str] = {abbrev: team_key for abbrev, team_key in NHL_TEAM_ABBREVIATIONS.items()}
    for row in standings_payload.get("standings", []):
        abbrev_payload = row.get("teamAbbrev") or {}
        name_payload = row.get("teamName") or {}
        abbrev = _extract_nhl_team_display_name(abbrev_payload)
        team_name = _extract_nhl_team_display_name(name_payload)
        if abbrev and team_name:
            teams_by_abbrev[abbrev.upper()] = _normalize_goalie_team_key(team_name)

    goalies: dict[str, dict[str, object]] = dict(probable_goalies)
    for abbrev, team_key in teams_by_abbrev.items():
        if NHL_GOALIE_RATE_LIMITED:
            break
        cache_key = abbrev.upper()
        if cache_key in NHL_GOALIE_TEAM_STATS_CACHE:
            NHL_GOALIE_API_STATS["cache_hits"] += 1
            cached = NHL_GOALIE_TEAM_STATS_CACHE[cache_key]
            if cached:
                goalies.setdefault(team_key, {})
                goalies[team_key].update(cached)
            continue

        payload = _nhl_api_get_json(f"https://api-web.nhle.com/v1/club-stats/{cache_key}/now")
        if payload is None:
            if NHL_GOALIE_RATE_LIMITED:
                break
            NHL_GOALIE_TEAM_STATS_CACHE[cache_key] = {}
            continue

        goalie_rows = payload.get("goalies") or []
        save_pcts = []
        for goalie in goalie_rows:
            save_pct = pd.to_numeric(pd.Series([goalie.get("savePctg")]), errors="coerce").iloc[0]
            games_played = pd.to_numeric(pd.Series([goalie.get("gamesPlayed")]), errors="coerce").iloc[0]
            if pd.notna(save_pct) and save_pct > 0 and pd.notna(games_played) and games_played > 0:
                save_pcts.append((float(games_played), float(save_pct)))
        if not save_pcts:
            NHL_GOALIE_TEAM_STATS_CACHE[cache_key] = {}
            continue
        total_games = sum(games for games, _ in save_pcts)
        weighted_save_pct = sum(games * save_pct for games, save_pct in save_pcts) / total_games
        team_values = {"save_pct": weighted_save_pct, "team_save_pct": weighted_save_pct, "source": "nhl_api_club_stats"}
        NHL_GOALIE_TEAM_STATS_CACHE[cache_key] = team_values
        goalies.setdefault(team_key, {})
        goalies[team_key].update(team_values)
    return goalies

def load_nhl_goalies() -> dict[str, dict[str, float]]:
    goalies = _fetch_nhl_goalies_from_api()

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
            for team, values in payload.items():
                if isinstance(values, dict):
                    goalies.setdefault(_normalize_goalie_team_key(team), dict(values))
            break
    return goalies

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
    stats["team_norm"] = stats["team_norm"].apply(clean_team_name).apply(_normalize_team)
    stats["team_key"] = stats["team_norm"].apply(_normalize_team_key).replace(TEAM_MAP)

    if "home_team" in out.columns:
        out["home_team"] = out["home_team"].apply(clean_team_name)
    if "away_team" in out.columns:
        out["away_team"] = out["away_team"].apply(clean_team_name)

    if "home_team_norm" not in out.columns and "home_team" in out.columns:
        out["home_team_norm"] = out["home_team"].apply(clean_team_name).apply(_normalize_team)
    if "away_team_norm" not in out.columns and "away_team" in out.columns:
        out["away_team_norm"] = out["away_team"].apply(clean_team_name).apply(_normalize_team)

    if "home_team_norm" in out.columns:
        out["home_team_norm"] = out["home_team_norm"].apply(clean_team_name).apply(_normalize_team)
    if "away_team_norm" in out.columns:
        out["away_team_norm"] = out["away_team_norm"].apply(clean_team_name).apply(_normalize_team)

    if "home_team" in out.columns:
        out["home_team_key"] = out["home_team"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
    else:
        out["home_team_key"] = ""
    if "away_team" in out.columns:
        out["away_team_key"] = out["away_team"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
    else:
        out["away_team_key"] = ""

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

    out = out.merge(home_stats, left_on="home_team_key", right_on="team_key_home", how="left")
    out = out.merge(away_stats, left_on="away_team_key", right_on="team_key_away", how="left")

    print("[MERGE SUCCESS CHECK]")
    if {"home_team", "offensive_rating_home", "offensive_rating_away"}.issubset(out.columns):
        print(out[["home_team", "offensive_rating_home", "offensive_rating_away"]].head())
    elif {"home_team", "goalie_save_strength_home", "goalie_save_strength_away"}.issubset(out.columns):
        print(out[["home_team", "goalie_save_strength_home", "goalie_save_strength_away"]].head())
    elif {"home_team", "starter_rating_home", "starter_rating_away"}.issubset(out.columns):
        print(out[["home_team", "starter_rating_home", "starter_rating_away"]].head())
        missing = out["starter_rating_home"].isna().sum()
        if missing > 0:
            print("⚠️ MLB merge missing values:", missing)
        out["starter_rating_home"] = out["starter_rating_home"].fillna(0)
        out["starter_rating_away"] = out["starter_rating_away"].fillna(0)

    if "home_team_key" in out.columns and "team_key" in stats.columns:
        missing = set(out["home_team_key"]) - set(stats["team_key"])
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
        if "home_team" in df.columns:
            df["home_team_norm"] = df["home_team"].apply(_normalize_goalie_team_key)
        if "away_team" in df.columns:
            df["away_team_norm"] = df["away_team"].apply(_normalize_goalie_team_key)
        df = enrich_nhl_live_features(df, nhl_team_stats=None)
        goalies = load_nhl_goalies()
        goalie_keys = set(goalies.keys())
        stats_keys = set()
        if team_df is not None and "team" in team_df.columns:
            stats_keys = set(team_df["team"].apply(_normalize_goalie_team_key).dropna().astype(str))

        print("[NHL GOALIE PARSE DEBUG]")
        print(f"goalie_records_parsed: {len(goalies) if isinstance(goalies, dict) else 0}")
        sample_goalies = [
            {"team_key": key, **(value if isinstance(value, dict) else {})}
            for key, value in list((goalies or {}).items())[:5]
        ]
        print("sample_parsed_goalie_records:", sample_goalies)
        print("goalie_map_team_keys:", sorted(goalie_keys)[:40])

        def _side_goalie_value(team_key: object) -> tuple[float, str, str, int, str]:
            normalized_key = str(team_key)
            record = goalies.get(normalized_key, {}) if isinstance(goalies, dict) else {}
            raw_save = record.get("save_pct", record.get("team_save_pct", np.nan)) if isinstance(record, dict) else np.nan
            save_pct = pd.to_numeric(pd.Series([raw_save]), errors="coerce").iloc[0]
            goalie_name = str(record.get("goalie", "")) if isinstance(record, dict) else ""
            source = str(record.get("source", "")) if isinstance(record, dict) else ""
            matched_key = normalized_key if normalized_key in goalie_keys else ""
            if pd.notna(save_pct) and float(save_pct) > 0:
                source_type = "real" if goalie_name and "probable" in source else "team_fallback"
                return float(save_pct), goalie_name, source_type, 0, matched_key
            return NHL_NEUTRAL_GOALIE_SAVE_PCT, "", "neutral", 0, matched_key

        home_rows = df["home_team_norm"].map(_side_goalie_value)
        away_rows = df["away_team_norm"].map(_side_goalie_value)
        df["goalie_save_home"] = home_rows.map(lambda item: item[0]).astype(float)
        df["goalie_save_away"] = away_rows.map(lambda item: item[0]).astype(float)
        df["goalie_home"] = home_rows.map(lambda item: item[1])
        df["goalie_away"] = away_rows.map(lambda item: item[1])
        df["goalie_source_home"] = home_rows.map(lambda item: item[2])
        df["goalie_source_away"] = away_rows.map(lambda item: item[2])
        df["goalie_home_source"] = df["goalie_source_home"]
        df["goalie_away_source"] = df["goalie_source_away"]
        df["goalie_home_team_key_used"] = home_rows.map(lambda item: item[4])
        df["goalie_away_team_key_used"] = away_rows.map(lambda item: item[4])
        df["starting_goalie_out_flag_home"] = home_rows.map(lambda item: item[3]).astype(int)
        df["starting_goalie_out_flag_away"] = away_rows.map(lambda item: item[3]).astype(int)

        different_teams = df["home_team"].astype(str).ne(df["away_team"].astype(str))
        same_goalie_mask = (
            df["goalie_home"].astype(str).str.strip().ne("")
            & df["goalie_home"].astype(str).str.strip().eq(df["goalie_away"].astype(str).str.strip())
            & different_teams
        )
        same_key_mask = (
            df["goalie_home_team_key_used"].astype(str).str.strip().ne("")
            & df["goalie_home_team_key_used"].astype(str).str.strip().eq(df["goalie_away_team_key_used"].astype(str).str.strip())
            & different_teams
        )
        invalid_goalie_mask = same_goalie_mask | same_key_mask
        invalid_goalie_assignments = int(invalid_goalie_mask.sum())
        if invalid_goalie_assignments:
            print(
                f"[NHL GOALIE WARNING] {invalid_goalie_assignments} games had invalid same-goalie "
                "or same-team-key assignments; using neutral goalie defaults for those games"
            )
            df.loc[invalid_goalie_mask, ["goalie_save_home", "goalie_save_away"]] = NHL_NEUTRAL_GOALIE_SAVE_PCT
            df.loc[invalid_goalie_mask, ["goalie_home", "goalie_away"]] = ""
            df.loc[
                invalid_goalie_mask,
                ["goalie_source_home", "goalie_source_away", "goalie_home_source", "goalie_away_source"],
            ] = "neutral_invalid"
            df.loc[invalid_goalie_mask, ["goalie_home_team_key_used", "goalie_away_team_key_used"]] = ""

        unmatched_home = sorted(set(df.loc[df["goalie_home_team_key_used"].eq(""), "home_team_norm"].dropna().astype(str)))
        unmatched_away = sorted(set(df.loc[df["goalie_away_team_key_used"].eq(""), "away_team_norm"].dropna().astype(str)))
        print("unmatched_home_teams:", unmatched_home[:40])
        print("unmatched_away_teams:", unmatched_away[:40])

        real_home = int((df["goalie_source_home"] == "real").sum())
        real_away = int((df["goalie_source_away"] == "real").sum())
        team_fallback_home = int((df["goalie_source_home"] == "team_fallback").sum())
        team_fallback_away = int((df["goalie_source_away"] == "team_fallback").sum())
        neutral_defaults = int(
            df["goalie_source_home"].astype(str).str.startswith("neutral").sum()
            + df["goalie_source_away"].astype(str).str.startswith("neutral").sum()
        )
        covered_sides = real_home + real_away + team_fallback_home + team_fallback_away
        coverage_pct = (covered_sides / (2 * len(df)) * 100.0) if len(df) else 0.0
        if neutral_defaults == 2 * len(df) and coverage_pct < 50.0:
            quality = "severe"
        elif neutral_defaults or (real_home + real_away and team_fallback_home + team_fallback_away):
            quality = "degraded"
        else:
            quality = "normal"
        df["goalie_coverage_pct"] = coverage_pct
        df["nhl_goalie_coverage_pct"] = coverage_pct
        df["goalie_data_quality_status"] = quality
        df = build_nhl_diff_features(df)
        print("[NHL TEAM KEY CHECK]")
        key_check = df.assign(
            normalized_home_key=df["home_team_norm"],
            matched_goalie_key=df["home_team_norm"].where(df["home_team_norm"].isin(goalie_keys), ""),
            matched_stats_key=df["home_team_norm"].where(df["home_team_norm"].isin(stats_keys), ""),
        ).rename(columns={"home_team": "raw_home_team"})
        print(key_check[["raw_home_team", "normalized_home_key", "matched_goalie_key", "matched_stats_key"]].head().to_string(index=False))
        print("[NHL GOALIE COVERAGE]")
        print(f"total_games: {len(df)}")
        print(f"real_home_goalies: {real_home}")
        print(f"real_away_goalies: {real_away}")
        print(f"team_fallback_home: {team_fallback_home}")
        print(f"team_fallback_away: {team_fallback_away}")
        print(f"neutral_defaults: {neutral_defaults}")
        print(f"coverage_pct: {coverage_pct:.1f}")
        print(f"data_quality: {quality}")
        _print_nhl_goalie_api_status(coverage_pct)
        print("[NHL GOALIE MATCH SAMPLE]")
        goalie_sample_columns = [
            "home_team",
            "away_team",
            "goalie_home",
            "goalie_away",
            "goalie_home_source",
            "goalie_away_source",
            "goalie_home_team_key_used",
            "goalie_away_team_key_used",
            "goalie_save_home",
            "goalie_save_away",
            "goalie_diff",
        ]
        print(df[goalie_sample_columns].head().to_string(index=False))
        return df

    if sport == "mlb":
        print("\n🚨 MLB PIPELINE STARTED 🚨")
        from sports_betting.sports.mlb.features import enrich_mlb_live_features, build_mlb_features

        try:
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

            pitchers_df = get_mlb_starting_pitchers()
            if "home_team" in df.columns and "away_team" in df.columns:
                df["home_team_key"] = df["home_team"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
                df["away_team_key"] = df["away_team"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
            else:
                df["home_team_key"] = ""
                df["away_team_key"] = ""
            if pitchers_df is None or pitchers_df.empty:
                pitchers_df = pd.DataFrame(columns=["home_team_key", "away_team_key", "home_pitcher", "away_pitcher"])
            else:
                pitchers_df = pitchers_df.rename(columns={"home_team": "home_team_key", "away_team": "away_team_key"})
                if "home_team_key" not in pitchers_df.columns:
                    pitchers_df["home_team_key"] = ""
                if "away_team_key" not in pitchers_df.columns:
                    pitchers_df["away_team_key"] = ""
                pitchers_df["home_team_key"] = pitchers_df["home_team_key"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
                pitchers_df["away_team_key"] = pitchers_df["away_team_key"].astype(str).str.lower().str.strip().apply(_normalize_team_key).replace(TEAM_MAP)
            unmatched_home = sorted(set(df["home_team_key"]) - set(pitchers_df["home_team_key"])) if "home_team_key" in df.columns else []
            unmatched_away = sorted(set(df["away_team_key"]) - set(pitchers_df["away_team_key"])) if "away_team_key" in df.columns else []
            if unmatched_home:
                print("[MLB UNMATCHED HOME TEAMS]", unmatched_home[:20])
            if unmatched_away:
                print("[MLB UNMATCHED AWAY TEAMS]", unmatched_away[:20])
            df = df.merge(
                pitchers_df,
                on=["home_team_key", "away_team_key"],
                how="left",
            )
            if "home_team_key" not in df.columns:
                print("🚨 MLB MERGE FAILED — SKIPPING MLB")
                return pd.DataFrame()
            print("[MLB TEAM KEYS SAMPLE]")
            print(df[["home_team", "home_team_key"]].drop_duplicates().head())
            print("[MLB MERGE CHECK]")
            print(df[["home_team", "starter_rating_home"]].head())
            if "home_pitcher_y" in df.columns:
                df["home_pitcher"] = df["home_pitcher_y"]
                if "home_pitcher_x" in df.columns:
                    df["home_pitcher"] = df["home_pitcher"].where(df["home_pitcher"].notna(), df["home_pitcher_x"])
            if "away_pitcher_y" in df.columns:
                df["away_pitcher"] = df["away_pitcher_y"]
                if "away_pitcher_x" in df.columns:
                    df["away_pitcher"] = df["away_pitcher"].where(df["away_pitcher"].notna(), df["away_pitcher_x"])
            df = df.drop(columns=[c for c in ["home_pitcher_x", "home_pitcher_y", "away_pitcher_x", "away_pitcher_y"] if c in df.columns])
            df = _standardize_mlb_pitcher_name_columns(df)

            if "home_probable_pitcher" not in df.columns:
                df["home_probable_pitcher"] = df.get("home_probable_pitcher", "")
            if "away_probable_pitcher" not in df.columns:
                df["away_probable_pitcher"] = df.get("away_probable_pitcher", "")
            if "home_pitcher" not in df.columns:
                df["home_pitcher"] = None
            if "away_pitcher" not in df.columns:
                df["away_pitcher"] = None
            df["starting_pitcher_home"] = df["home_pitcher"]
            df["starting_pitcher_away"] = df["away_pitcher"]
            df["home_probable_pitcher"] = df["home_probable_pitcher"].astype(str).where(
                df["home_probable_pitcher"].astype(str).str.strip().ne(""),
                df["home_pitcher"].fillna("").astype(str),
            )
            df["away_probable_pitcher"] = df["away_probable_pitcher"].astype(str).where(
                df["away_probable_pitcher"].astype(str).str.strip().ne(""),
                df["away_pitcher"].fillna("").astype(str),
            )
            df = _standardize_mlb_pitcher_name_columns(df)

            for pitcher_col in ["pitcher_era_home", "pitcher_era_away", "pitcher_diff"]:
                if pitcher_col not in df.columns:
                    df[pitcher_col] = np.nan
            print("[MLB DEBUG] Checking pitcher columns...")
            print(df[["home_team", "away_team", "pitcher_era_home", "pitcher_era_away", "pitcher_diff"]].head())

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
            df = _standardize_mlb_pitcher_name_columns(df)
            df["pitcher_era_home"] = pd.to_numeric(home_pitcher_stats.map(lambda v: v.get("pitcher_era")), errors="coerce")
            df["pitcher_era_away"] = pd.to_numeric(away_pitcher_stats.map(lambda v: v.get("pitcher_era")), errors="coerce")
            df["pitcher_era_home_is_real"] = home_pitcher_stats.map(lambda v: bool(v.get("pitcher_era_is_real", False))) & df["pitcher_name_home"].str.strip().ne("") & df["pitcher_era_home"].map(_is_real_mlb_era_value)
            df["pitcher_era_away_is_real"] = away_pitcher_stats.map(lambda v: bool(v.get("pitcher_era_is_real", False))) & df["pitcher_name_away"].str.strip().ne("") & df["pitcher_era_away"].map(_is_real_mlb_era_value)
            pitcher_debug_rows = []
            for side, stats_series in (("home", home_pitcher_stats), ("away", away_pitcher_stats)):
                name_col = f"pitcher_name_{side}"
                real_col = f"pitcher_era_{side}_is_real"
                for idx in df.index:
                    pitcher_name = df.at[idx, name_col]
                    pitcher_debug_rows.append(
                        {
                            "side": side,
                            "pitcher_name": pitcher_name,
                            "normalized_pitcher_name": _normalize_pitcher_name(pitcher_name),
                            "era_matched": bool(df.at[idx, real_col]),
                            "era_source": stats_series.loc[idx].get("pitcher_era_source", "unmatched"),
                        }
                    )
            print("[MLB PITCHER ERA DEBUG]")
            print(pd.DataFrame(pitcher_debug_rows).to_string(index=False))
            total_mlb_games = int(len(df))
            real_home_era_count = int(df["pitcher_era_home_is_real"].sum())
            real_away_era_count = int(df["pitcher_era_away_is_real"].sum())
            real_both_era_count = int((df["pitcher_era_home_is_real"] & df["pitcher_era_away_is_real"]).sum())
            real_pitcher_coverage_pct = (real_both_era_count / total_mlb_games * 100.0) if total_mlb_games else 0.0
            df["real_home_era_count"] = real_home_era_count
            df["real_away_era_count"] = real_away_era_count
            df["real_both_era_count"] = real_both_era_count
            df["real_pitcher_coverage_pct"] = real_pitcher_coverage_pct
            df["mlb_pitcher_coverage_pct"] = real_pitcher_coverage_pct
            df["pitcher_coverage_pct"] = real_pitcher_coverage_pct
            df["data_quality_status"] = _mlb_quality_from_real_coverage(real_pitcher_coverage_pct)
            default_era_count = int((~df["pitcher_era_home_is_real"]).sum() + (~df["pitcher_era_away_is_real"]).sum())
            df["default_era_count"] = default_era_count
            if real_pitcher_coverage_pct < MLB_REAL_ERA_NORMAL_THRESHOLD and "confidence" in df.columns:
                df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5) * 0.75
            print("[MLB REAL PITCHER ERA COVERAGE]")
            print("total_games:", total_mlb_games)
            print("real both ERAs:", f"{real_both_era_count}/{total_mlb_games}")
            print("real home ERAs:", f"{real_home_era_count}/{total_mlb_games}")
            print("real away ERAs:", f"{real_away_era_count}/{total_mlb_games}")
            print("default ERA count:", default_era_count)
            print("real coverage pct:", f"{real_pitcher_coverage_pct:.1f}%")
            print("data quality:", df["data_quality_status"].iloc[0] if len(df) else "severe")
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

            df["pitcher_era_home"] = df["pitcher_era_home"].fillna(DEFAULT_MLB_ERA)
            df["pitcher_era_away"] = df["pitcher_era_away"].fillna(DEFAULT_MLB_ERA)
            df["pitcher_whip_home"] = df["pitcher_whip_home"].fillna((df["pitcher_era_home"] / 4.0).clip(lower=0.9, upper=1.8))
            df["pitcher_whip_away"] = df["pitcher_whip_away"].fillna((df["pitcher_era_away"] / 4.0).clip(lower=0.9, upper=1.8))
            df["pitcher_k_rate_home"] = df["pitcher_k_rate_home"].fillna((0.30 - (df["pitcher_era_home"] - 3.0) * 0.025).clip(lower=0.12, upper=0.35))
            df["pitcher_k_rate_away"] = df["pitcher_k_rate_away"].fillna((0.30 - (df["pitcher_era_away"] - 3.0) * 0.025).clip(lower=0.12, upper=0.35))
            df["pitcher_era_diff"] = df["pitcher_era_away"] - df["pitcher_era_home"]
            df["pitcher_diff"] = df["pitcher_era_diff"]
            df = _populate_mlb_starter_ratings_from_era(df)
            if df["pitcher_era_home"].isna().all() or (df["pitcher_era_home"] == 0).all():
                print("🚨 MLB PITCHERS NOT WORKING — ALL ZERO OR NULL")
            df["pitcher_whip_diff"] = df["pitcher_whip_away"] - df["pitcher_whip_home"]
            df["pitcher_k_rate_diff"] = df["pitcher_k_rate_home"] - df["pitcher_k_rate_away"]
            print("[MLB FIXED] Pitcher data sample:")
            print(df[["home_team", "away_team", "home_pitcher", "away_pitcher", "pitcher_diff"]].head())

            PITCHER_WEIGHT = 0.015
            if "edge" in df.columns:
                if "adjusted_edge" in df.columns:
                    df["adjusted_edge"] = pd.to_numeric(df["adjusted_edge"], errors="coerce")
                    df["adjusted_edge"] = df["adjusted_edge"].where(df["adjusted_edge"].notna(), pd.to_numeric(df["edge"], errors="coerce") + (df["pitcher_diff"] * PITCHER_WEIGHT))
                else:
                    df["adjusted_edge"] = pd.to_numeric(df["edge"], errors="coerce") + (df["pitcher_diff"] * PITCHER_WEIGHT)
            df = enrich_mlb_live_features(df)
            df = _populate_mlb_starter_ratings_from_era(df)
            df["starter_rating_home"] = pd.to_numeric(df["starter_rating_home"], errors="coerce").fillna(50)
            df["starter_rating_away"] = pd.to_numeric(df["starter_rating_away"], errors="coerce").fillna(50)
            df["hitting_rating_home"] = pd.to_numeric(df["hitting_rating_home"], errors="coerce").replace(0, 100).fillna(100)
            df["hitting_rating_away"] = pd.to_numeric(df["hitting_rating_away"], errors="coerce").replace(0, 100).fillna(100)
            df["starter_rating_diff"] = df["starter_rating_home"] - df["starter_rating_away"]
            df["hitting_rating_diff"] = df["hitting_rating_home"] - df["hitting_rating_away"]
            print("Non-zero starter_diff:", (df["starter_rating_diff"] != 0).sum())
            if "home_team_key" not in df.columns:
                print("🚨 MLB MERGE FAILED — SKIPPING MLB")
                return pd.DataFrame()
            df = build_mlb_features(df)
            pitcher_signal = pd.to_numeric(df.get("pitcher_diff", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).abs().sum()
            starter_signal = pd.to_numeric(df.get("starter_rating_diff", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).abs().sum()
            odds_loaded = {"home_odds", "away_odds"}.issubset(df.columns) and (pd.to_numeric(df["home_odds"], errors="coerce").fillna(0).ne(0).any())
            if not odds_loaded:
                reason = f"pitcher_diff signal={pitcher_signal:.3f}, starter_rating_diff signal={starter_signal:.3f}, odds_loaded={odds_loaded}"
                print(f"[MLB SKIP] Required odds unavailable — {reason}")
                return pd.DataFrame()
            if pitcher_signal == 0 and starter_signal == 0 and real_pitcher_coverage_pct >= MLB_REAL_ERA_NORMAL_THRESHOLD:
                reason = f"pitcher_diff signal={pitcher_signal:.3f}, starter_rating_diff signal={starter_signal:.3f}, odds_loaded={odds_loaded}"
                print(f"[MLB SKIP] Required starter/pitcher signals unavailable — {reason}")
                return pd.DataFrame()
            return df
        except Exception as exc:
            print(f"[MLB PIPELINE ERROR] {exc}")
            print(traceback.format_exc())
            if "starter_rating_home" in df.columns:
                df["starter_rating_home"] = pd.to_numeric(df["starter_rating_home"], errors="coerce").fillna(0)
            else:
                df["starter_rating_home"] = 0
            if "starter_rating_away" in df.columns:
                df["starter_rating_away"] = pd.to_numeric(df["starter_rating_away"], errors="coerce").fillna(0)
            else:
                df["starter_rating_away"] = 0
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
