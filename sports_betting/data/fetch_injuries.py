import json
import pandas as pd
from pathlib import Path
from urllib.request import urlopen

STAR_PLAYER_MULTIPLIER = 2.0
ROLE_MULTIPLIERS = {
    "superstar": 2.0,
    "all-star": 1.75,
    "all star": 1.75,
    "star": 1.6,
    "starter": 1.35,
    "rotation": 1.15,
    "sixth man": 1.15,
    "bench": 1.0,
    "depth": 0.9,
}
STAR_PLAYER_BY_SPORT = {
    "nba": {
        "lebron james",
        "nikola jokic",
        "giannis antetokounmpo",
        "luka doncic",
        "jayson tatum",
        "stephen curry",
        "kevin durant",
        "joel embiid",
        "shai gilgeous alexander",
        "anthony davis",
    },
    "nhl": {
        "connor mcdavid",
        "nathan mackinnon",
        "auston matthews",
        "nikita kucherov",
        "david pastrnak",
        "cale makar",
        "igor shesterkin",
        "andrei vasilevskiy",
        "ilya sorokin",
    },
}


def _normalize_team_name(name: str | None) -> str:
    return normalize_team_name(str(name or ""))


def normalize_team_name(name):
    if not isinstance(name, str):
        return ""
    return (
        name.lower()
        .replace("trail blazers", "blazers")
        .replace("76ers", "sixers")
        .replace("la clippers", "los angeles clippers")
        .replace("la lakers", "los angeles lakers")
        .replace("okc thunder", "oklahoma city thunder")
        .replace("utah mammoth", "utah hockey club")
        .replace("montréal", "montreal")
        .strip()
    )


def _team_tokens(team_name: str | None) -> set[str]:
    return set(_normalize_team_name(team_name).split())


def _expand_city_aliases(team_name: str | None) -> str:
    aliases = {"la": "los angeles", "ny": "new york", "st": "saint"}
    expanded: list[str] = []
    for token in _normalize_team_name(team_name).split():
        alias = aliases.get(token)
        if alias:
            expanded.extend(alias.split())
        else:
            expanded.append(token)
    return " ".join(expanded).strip()


def _resolve_team_key(team_name: str | None, available_keys: set[str]) -> str | None:
    normalized = _normalize_team_name(team_name)
    if not normalized:
        return None
    if normalized in available_keys:
        return normalized

    incoming_tokens = _team_tokens(_expand_city_aliases(normalized))
    if not incoming_tokens:
        return None

    best_match = None
    best_overlap = 0.0
    for candidate in available_keys:
        candidate_tokens = _team_tokens(_expand_city_aliases(candidate))
        if not candidate_tokens:
            continue
        overlap = len(incoming_tokens & candidate_tokens) / max(len(incoming_tokens), len(candidate_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = candidate
    return best_match if best_overlap >= 0.5 else None


def _status_weight(status: str | None) -> float:
    normalized = str(status or "").strip().lower()
    if normalized in {"out", "inactive", "ir"}:
        return 1.0
    if normalized == "doubtful":
        return 0.75
    if normalized in {"questionable", "day-to-day", "day to day"}:
        return 0.5
    if normalized == "probable":
        return 0.25
    return 0.4


def fetch_espn_injuries_for_sport(sport: str) -> pd.DataFrame:
    sport_key = str(sport or "").strip().lower()
    endpoints = {
        "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
        "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
    }
    if sport_key not in endpoints:
        raise ValueError(f"Unsupported sport for injury fetch: {sport}")
    url = endpoints[sport_key]
    with urlopen(url, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    injuries = []
    for team in data.get("teams", []):
        team_name = team.get("team", {}).get("displayName")
        for athlete in team.get("injuries", []):
            player_name = athlete.get("athlete", {}).get("displayName")
            status = athlete.get("status")
            injuries.append(
                {
                    "sport": sport_key,
                    "team": _normalize_team_name(team_name),
                    "player": player_name,
                    "status": status,
                }
            )

    df = pd.DataFrame(injuries)
    if df.empty:
        return pd.DataFrame(columns=["sport", "team", "player", "status"])
    return df


def fetch_nba_injuries() -> pd.DataFrame:
    return fetch_espn_injuries_for_sport("nba")


def fetch_nhl_injuries() -> pd.DataFrame:
    return fetch_espn_injuries_for_sport("nhl")


def fetch_injuries(sport: str) -> pd.DataFrame:
    return fetch_espn_injuries_for_sport(sport)


def _ensure_injury_frame(df_injuries: pd.DataFrame) -> pd.DataFrame:
    if df_injuries.empty:
        return pd.DataFrame(columns=["sport", "team", "player", "status"])
    injuries = df_injuries.copy()
    if "team_name" in injuries.columns and "team" not in injuries.columns:
        injuries["team"] = injuries["team_name"]
    for col in ["sport", "team", "player", "status"]:
        if col not in injuries.columns:
            injuries[col] = ""
    injuries["team"] = injuries["team"].astype(str).apply(_normalize_team_name)
    injuries["status"] = injuries["status"].astype(str)
    return injuries


def _is_star_player(player_name: str, sport: str, role: str) -> bool:
    player_key = str(player_name or "").strip().lower()
    role_key = str(role or "").strip().lower()
    if not player_key:
        return False
    if player_key in STAR_PLAYER_BY_SPORT.get(str(sport or "").strip().lower(), set()):
        return True
    return role_key in {"superstar", "all-star", "all star", "star", "elite qb", "starting goalie"}


def _player_injury_impact(row: pd.Series) -> float:
    status_weight = _status_weight(row.get("status"))
    role_key = str(row.get("role") or row.get("expected_minutes_or_role") or "").strip().lower()
    position_key = str(row.get("position") or "").strip().lower()
    role_multiplier = ROLE_MULTIPLIERS.get(role_key, 1.0)

    expected_minutes = pd.to_numeric(row.get("expected_minutes") or row.get("minutes") or row.get("expected_minutes_or_role"), errors="coerce")
    if pd.notna(expected_minutes):
        if expected_minutes >= 34:
            role_multiplier = max(role_multiplier, 1.6)
        elif expected_minutes >= 28:
            role_multiplier = max(role_multiplier, 1.35)
        elif expected_minutes >= 20:
            role_multiplier = max(role_multiplier, 1.15)
    if position_key in {"qb", "g", "goalie"}:
        role_multiplier = max(role_multiplier, 1.5)

    star_multiplier = STAR_PLAYER_MULTIPLIER if _is_star_player(row.get("player"), row.get("sport"), role_key) else 1.0
    return status_weight * role_multiplier * star_multiplier


def compute_injury_impact(df_games: pd.DataFrame, df_injuries: pd.DataFrame) -> pd.DataFrame:
    out = df_games.copy()
    injuries_path = Path("sports_betting/data/injuries/injuries.json")
    if not injuries_path.exists():
        injuries_path = Path(__file__).resolve().parent / "injuries" / "injuries.json"

    if injuries_path.exists():
        with injuries_path.open(encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}
        if isinstance(df_injuries, pd.DataFrame) and not df_injuries.empty:
            team_col = "team" if "team" in df_injuries.columns else "team_name"
            player_col = "player" if "player" in df_injuries.columns else "player_name"
            if team_col in df_injuries.columns and player_col in df_injuries.columns:
                for _, rec in df_injuries.iterrows():
                    team = str(rec.get(team_col, "")).strip()
                    player = str(rec.get(player_col, "")).strip()
                    if not team or not player:
                        continue
                    raw.setdefault(team, []).append(player)

    rows = []
    for team, players in raw.items():
        for player in players:
            rows.append(
                {
                    "team": team,
                    "player": player,
                    "status": "out",
                }
            )

    injuries = pd.DataFrame(rows)

    if injuries.empty:
        print("[ERROR] Injuries dataframe is empty after rebuild")

    if "home_team" not in out.columns:
        out["home_team"] = ""
    if "away_team" not in out.columns:
        out["away_team"] = ""

    injuries["team_norm"] = injuries["team"].apply(normalize_team_name)
    out["home_team_norm"] = out["home_team"].apply(normalize_team_name)
    out["away_team_norm"] = out["away_team"].apply(normalize_team_name)

    injury_counts = injuries.groupby("team_norm").size()

    out["injury_impact_home"] = out["home_team_norm"].map(injury_counts).fillna(0)
    out["injury_impact_away"] = out["away_team_norm"].map(injury_counts).fillna(0)
    out["injury_impact_diff"] = out["injury_impact_home"] - out["injury_impact_away"]

    print("\n[INJURY RAW SAMPLE]:", list(raw.keys())[:5])
    print("\n[INJURY DF SAMPLE]:")
    print(injuries.head())
    print("\n[TEAM NORMALIZATION SAMPLE]:")
    print(out[["home_team", "home_team_norm"]].head())
    print("\n[INJURY MATCH COUNT]:", (out["injury_impact_diff"] != 0).sum())

    return out
