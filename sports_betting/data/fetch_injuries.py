import json
import pandas as pd
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from sports_betting.sports.common.team_names import normalize_team_name as shared_normalize_team_name

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
SUPERSTAR_WEIGHT = 2.5
STARTER_WEIGHT = 1.5
ROLE_PLAYER_WEIGHT = 1.0


def clean_team_name(name: str | None) -> str:
    return (
        str(name or "").lower()
        .strip()
        .replace(".", "")
        .replace("-", " ")
    )


def normalize_team(name: object) -> str:
    return str(name or "").lower().strip()


def _normalize_team_name(name: str | None) -> str:
    return normalize_team_name(clean_team_name(str(name or "")))


def normalize_team_name(name):
    if not isinstance(name, str):
        return ""
    normalized = str(shared_normalize_team_name(name)).strip()
    replacements = {
        "portland blazers": "portland trail blazers",
        "st louis blues": "st louis blues",
        "utah hockey club": "utah mammoth",
    }
    return replacements.get(normalized, normalized)


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


def classify_player(player_name: str, role: str | None = None) -> float:
    superstar_list = [
        "lebron james",
        "stephen curry",
        "nikola jokic",
        "giannis antetokounmpo",
        "luka doncic",
        "kevin durant",
    ]

    normalized_name = str(player_name or "").strip().lower()
    if normalized_name in superstar_list:
        return SUPERSTAR_WEIGHT

    role_key = str(role or "").strip().lower()
    if role_key in {"bench", "depth", "role player"}:
        return ROLE_PLAYER_WEIGHT
    if role_key == "rotation":
        return 1.2

    return STARTER_WEIGHT  # fallback for now


def fetch_espn_injuries_for_sport(sport: str) -> pd.DataFrame:
    sport_key = str(sport or "").strip().lower()
    endpoints = {
        "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
        "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
    }
    if sport_key not in endpoints:
        raise ValueError(f"Unsupported sport for injury fetch: {sport}")
    url = endpoints[sport_key]
    print(f"[INJURY DEBUG] Fetching {sport_key} injuries from: {url}")
    fetch_success = False
    fallback_used = False
    injuries = []
    # Try multiple request approaches to handle ESPN's protection
    request_attempts = [
        # Standard request with sports-specific user agent
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36', 'Accept': 'application/json,*/*', 'Referer': 'https://www.espn.com/'},
        # Mobile user agent
        {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'},
        # Original simple approach
        {'User-Agent': 'Mozilla/5.0 (compatible; SportsBetting/1.0)'}
    ]
    
    fetch_success = False
    data = {}
    last_exception = None
    
    for i, headers in enumerate(request_attempts):
        try:
            print(f"[INJURY DEBUG] Attempt {i+1}: Using User-Agent: {headers['User-Agent'][:50]}...")
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                response_text = response.read().decode("utf-8")
                print(f"[INJURY DEBUG] Response status: {response.status}")
                print(f"[INJURY DEBUG] Raw response length: {len(response_text)} characters")
                
                # Check if we got HTML instead of JSON (redirect/error page)
                if response_text.strip().startswith('<'):
                    print(f"[INJURY DEBUG] Got HTML response instead of JSON - likely blocked or redirected")
                    print(f"[INJURY DEBUG] HTML preview: {response_text[:200]}...")
                    continue
                
                data = json.loads(response_text)
                fetch_success = True
                print(f"[INJURY DEBUG] JSON parsed successfully. Keys: {list(data.keys())}")
                break
                
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_exception = exc
            print(f"[INJURY DEBUG] Attempt {i+1} failed: {type(exc).__name__}: {exc}")
            continue
    
    if not fetch_success:
        print(f"[INJURY WARNING] All ESPN injury API attempts failed for {sport_key}. Last error: {last_exception}")
    
    teams_data = data.get("teams", [])
    print(f"[INJURY DEBUG] Found {len(teams_data)} teams in response")
    
    # If no teams found, check if data structure has changed
    if not teams_data and data:
        print(f"[INJURY DEBUG] No 'teams' key found. Checking alternative structures...")
        # Check for alternative data structures
        possible_keys = ['items', 'data', 'results', 'leagues', 'competitions']
        for key in possible_keys:
            if key in data:
                alt_data = data[key]
                print(f"[INJURY DEBUG] Found alternative key '{key}' with {len(alt_data) if isinstance(alt_data, list) else 'non-list'} items")
                if isinstance(alt_data, list) and alt_data:
                    print(f"[INJURY DEBUG] Sample item keys: {list(alt_data[0].keys()) if isinstance(alt_data[0], dict) else 'not dict'}")
    
    for i, team in enumerate(teams_data):
        team_name = team.get("team", {}).get("displayName")
        team_injuries = team.get("injuries", [])
        print(f"[INJURY DEBUG] Team {i+1}: {team_name} has {len(team_injuries)} injuries")
        
        for j, athlete in enumerate(team_injuries):
            # Handle different possible data structures for athlete info
            if isinstance(athlete, dict):
                player_name = athlete.get("athlete", {}).get("displayName") or athlete.get("displayName") or athlete.get("name")
                status = athlete.get("status") or athlete.get("injuryStatus") or "out"
            else:
                # Fallback if athlete is not a dict
                player_name = str(athlete) if athlete else None
                status = "out"
            
            print(f"[INJURY DEBUG]   Player {j+1}: {player_name} - {status}")
            
            if player_name:  # Only add if we have a player name
                injuries.append(
                    {
                        "sport": sport_key,
                        "team": _normalize_team_name(team_name),
                        "player": player_name,
                        "status": status,
                    }
                )

    print(f"[INJURY DEBUG] Total injuries collected: {len(injuries)}")
    
    if sport_key == "nba" and fetch_success and not injuries:
        print(f"[INJURY DEBUG] No injuries from main API, trying fallback scoreboard...")
        fallback_used = True
        fallback_injuries = _fetch_nba_espn_scoreboard_injuries()
        print(f"[INJURY DEBUG] Scoreboard fallback found {len(fallback_injuries)} injuries")
        injuries.extend(fallback_injuries)
        
        # If scoreboard also failed, try alternative sources
        if not fallback_injuries:
            print(f"[INJURY DEBUG] Scoreboard fallback also empty, trying alternative sources...")
            try:
                from .injuries.alternative_sources import get_fallback_injury_data
                alt_data = get_fallback_injury_data()
                if alt_data:
                    print(f"[INJURY DEBUG] Alternative sources found {len(alt_data)} teams")
                    # Convert alternative data format to our format
                    for team_name, players in alt_data.items():
                        for player_name, status in players.items():
                            injuries.append({
                                "sport": sport_key,
                                "team": _normalize_team_name(team_name),
                                "player": player_name,
                                "status": status,
                            })
                    print(f"[INJURY DEBUG] Added {len(alt_data)} teams from alternative sources")
            except Exception as e:
                print(f"[INJURY DEBUG] Alternative sources failed: {e}")

    df = pd.DataFrame(injuries)
    if df.empty:
        df = pd.DataFrame(columns=["sport", "team", "player", "status"])
    teams_found = int(df["team"].nunique()) if "team" in df.columns else 0
    status = "normal" if len(df) else "degraded"
    df.attrs["source"] = "espn_api" if not fallback_used else "espn_scoreboard_fallback"
    df.attrs["fetch_success"] = fetch_success
    df.attrs["rows_parsed"] = int(len(df))
    df.attrs["teams_found"] = teams_found
    df.attrs["fallback_used"] = fallback_used
    df.attrs["data_quality_status"] = status
    print("[INJURY SOURCE STATUS]")
    print(f"source: {df.attrs['source']}")
    print(f"fetch_success: {fetch_success}")
    print(f"rows_parsed: {len(df)}")
    print(f"teams_found: {teams_found}")
    print(f"fallback_used: {fallback_used}")
    print(f"data_quality_status: {status}")
    return df


def _fetch_nba_espn_scoreboard_injuries() -> list[dict[str, object]]:
    """Fallback parser for injuries embedded in ESPN's NBA scoreboard payload."""
    rows: list[dict[str, object]] = []
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    print(f"[INJURY DEBUG] Attempting scoreboard fallback: {scoreboard_url}")
    try:
        request = Request(
            scoreboard_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SportsBetting/1.0)"},
        )
        with urlopen(request, timeout=20) as response:
            response_text = response.read().decode("utf-8")
            print(f"[INJURY DEBUG] Scoreboard response length: {len(response_text)} characters")
            payload = json.loads(response_text)
        print(f"[INJURY DEBUG] Scoreboard JSON keys: {list(payload.keys())}")
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"[INJURY WARNING] ESPN scoreboard fallback failed for nba: {exc}")
        print(f"[INJURY DEBUG] Scoreboard exception type: {type(exc).__name__}")
        return rows

    events = payload.get("events", [])
    print(f"[INJURY DEBUG] Found {len(events)} events in scoreboard")
    
    for i, event in enumerate(events):
        competitions = event.get("competitions", [])
        print(f"[INJURY DEBUG] Event {i+1} has {len(competitions)} competitions")
        
        for j, competition in enumerate(competitions):
            competitors = competition.get("competitors", [])
            print(f"[INJURY DEBUG]   Competition {j+1} has {len(competitors)} competitors")
            
            for k, competitor in enumerate(competitors):
                team_name = competitor.get("team", {}).get("displayName")
                competitor_injuries = competitor.get("injuries", []) or []
                print(f"[INJURY DEBUG]     Competitor {k+1}: {team_name} has {len(competitor_injuries)} injuries")
                
                for l, injury in enumerate(competitor_injuries):
                    athlete = injury.get("athlete", {}) if isinstance(injury, dict) else {}
                    player = athlete.get("displayName") or injury.get("displayName") or injury.get("name")
                    status = injury.get("status") or injury.get("type") or "out"
                    print(f"[INJURY DEBUG]       Injury {l+1}: {player} - {status}")
                    
                    if not player:
                        continue
                    rows.append(
                        {
                            "sport": "nba",
                            "team": _normalize_team_name(team_name),
                            "player": player,
                            "status": status,
                        }
                    )
    return rows


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
    injury_status = getattr(df_injuries, "attrs", {}) if isinstance(df_injuries, pd.DataFrame) else {}
    injury_rows_parsed = int(injury_status.get("rows_parsed", len(df_injuries) if isinstance(df_injuries, pd.DataFrame) else 0))
    injury_degraded = injury_rows_parsed == 0 or str(injury_status.get("data_quality_status", "normal")) != "normal"
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
                    team = clean_team_name(str(rec.get(team_col, "")).strip())
                    player = str(rec.get(player_col, "")).strip()
                    if not team or not player:
                        continue
                    raw.setdefault(team, []).append(
                        {
                            "player": player,
                            "status": rec.get("status", "out"),
                            "role": rec.get("role") or rec.get("expected_minutes_or_role", ""),
                        }
                    )

    rows = []
    for team, players in raw.items():
        for player in players:
            player_name = player
            status = "out"
            role = ""
            if isinstance(player, dict):
                player_name = player.get("player") or player.get("player_name") or ""
                status = player.get("status", "out")
                role = player.get("role") or player.get("expected_minutes_or_role", "")
            rows.append(
                {
                    "team": team,
                    "player": player_name,
                    "status": status,
                    "role": role,
                }
            )

    injuries = pd.DataFrame(rows, columns=["team", "player", "status", "role"])

    if injuries.empty:
        print("[INJURY INFO] No parsed injuries available; using neutral injury features with degraded status")
        out["injury_impact_home"] = 0.0
        out["injury_impact_away"] = 0.0
        out["injury_impact_diff"] = 0.0
        out["injury_data_stale_flag"] = 1
        out["injury_confidence_score"] = 0.0
        out["injury_data_quality_status"] = "degraded"
        return out

    if "home_team" not in out.columns:
        out["home_team"] = ""
    if "away_team" not in out.columns:
        out["away_team"] = ""

    injuries["team_clean"] = injuries["team"].apply(normalize_team)
    out["home_team_clean"] = out["home_team"].apply(normalize_team)
    out["away_team_clean"] = out["away_team"].apply(normalize_team)

    injuries["team_norm"] = injuries["team_clean"].apply(normalize_team_name)
    out["home_team_norm"] = out["home_team_clean"].apply(normalize_team_name)
    out["away_team_norm"] = out["away_team_clean"].apply(normalize_team_name)
    print("INJURY TEAMS:", injuries["team"].dropna().astype(str).unique())
    print("MODEL TEAMS:", out["home_team"].dropna().astype(str).unique())

    injuries["player_weight"] = injuries.apply(
        lambda injury_row: classify_player(
            injury_row.get("player", ""),
            injury_row.get("role") or injury_row.get("expected_minutes_or_role"),
        ),
        axis=1,
    )
    print("[PLAYER WEIGHT DEBUG]")
    for _, player_row in injuries.iterrows():
        player_name = player_row.get("player", "")
        weight = float(player_row.get("player_weight", STARTER_WEIGHT))
        print(player_name, "weight:", weight)
    injury_counts = injuries.groupby("team_norm")["player_weight"].sum()
    available_injury_teams = set(injury_counts.index)
    out["home_team_match"] = out["home_team_norm"].apply(lambda team: _resolve_team_key(team, available_injury_teams) or team)
    out["away_team_match"] = out["away_team_norm"].apply(lambda team: _resolve_team_key(team, available_injury_teams) or team)

    out["injury_impact_home"] = out["home_team_match"].map(injury_counts).fillna(0)
    out["injury_impact_away"] = out["away_team_match"].map(injury_counts).fillna(0)
    out["injury_impact_diff"] = out["injury_impact_home"] - out["injury_impact_away"]
    out["injury_data_stale_flag"] = int(injury_degraded)
    out["injury_confidence_score"] = 0.0 if injury_degraded else 1.0
    out["injury_data_quality_status"] = "degraded" if injury_degraded else "normal"

    print("\n[INJURY RAW SAMPLE]:", list(raw.keys())[:5])
    print("\n[INJURY DF SAMPLE]:")
    print(injuries.head())
    print("\n[TEAM NORMALIZATION SAMPLE]:")
    print(out[["home_team", "home_team_norm"]].head())
    print("\n[INJURY MATCH COUNT]:", (out["injury_impact_diff"] != 0).sum())

    out = out.drop(columns=["home_team_clean", "away_team_clean"], errors="ignore")
    return out
