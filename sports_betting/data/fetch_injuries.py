import json
import pandas as pd
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
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(name)).split())


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
    """Refined injury impact calculation for teams based on player injuries."""
    out = df_games.copy()
    injuries = _ensure_injury_frame(df_injuries)

    if "home_team" not in out.columns:
        out["home_team"] = ""
    if "away_team" not in out.columns:
        out["away_team"] = ""

    # Keep display team names intact, but create normalized mapping keys.
    out["_home_team_key"] = out["home_team"].astype(str).apply(_normalize_team_name)
    out["_away_team_key"] = out["away_team"].astype(str).apply(_normalize_team_name)
    if not injuries.empty:
        injuries["impact"] = injuries.apply(_player_injury_impact, axis=1)
    else:
        injuries["impact"] = pd.Series(dtype=float)

    # Weighted injury mapping to teams, keeps NHL/NBA status severity.
    injury_map = injuries.groupby("team")["impact"].sum()
    print(f"[INJURY IMPACT] teams with mapped injuries: {len(injury_map)}")

    available_keys = set(injury_map.index.astype(str))
    out["_home_injury_key"] = out["_home_team_key"].apply(lambda team: _resolve_team_key(team, available_keys))
    out["_away_injury_key"] = out["_away_team_key"].apply(lambda team: _resolve_team_key(team, available_keys))

    # Apply injury impact mapping to teams
    out["injury_impact_home"] = out["_home_injury_key"].map(injury_map).fillna(0)
    out["injury_impact_away"] = out["_away_injury_key"].map(injury_map).fillna(0)

    # Calculate the difference between away team and home team injury impact
    out["injury_impact_diff"] = out["injury_impact_away"] - out["injury_impact_home"]
    matched_games = ((out["injury_impact_home"] != 0) | (out["injury_impact_away"] != 0)).sum()
    print(f"[INJURY IMPACT] matched games: {int(matched_games)} / {len(out)}")
    if not injury_map.empty:
        print("[INJURY IMPACT] top impacted teams:")
        print(injury_map.sort_values(ascending=False).head(10))

    out = out.drop(columns=["_home_team_key", "_away_team_key", "_home_injury_key", "_away_injury_key"], errors="ignore")

    return out
