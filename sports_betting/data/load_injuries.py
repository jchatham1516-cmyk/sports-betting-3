import json
import os

from sports_betting.sports.common.team_names import normalize_team_name


def normalize_player_name(name):
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(name or "")).split())


def _team_tokens(team_name):
    return set(normalize_team_name(team_name).split())


CITY_ALIASES = {
    "la": "los angeles",
    "ny": "new york",
}


def _expand_city_aliases(team_name):
    tokens = normalize_team_name(team_name).split()
    expanded = []
    for token in tokens:
        alias = CITY_ALIASES.get(token)
        if alias:
            expanded.extend(alias.split())
        else:
            expanded.append(token)
    return " ".join(expanded).strip()


def resolve_injury_team_key(team_name, injuries):
    """Best-effort resolver for team names between odds/predictions and injury feeds."""
    normalized = normalize_team_name(team_name)
    if not normalized:
        return None
    if normalized in injuries:
        return normalized

    incoming_tokens = _team_tokens(_expand_city_aliases(normalized))
    if not incoming_tokens:
        return None

    best_match = None
    best_overlap = 0.0
    for candidate in injuries.keys():
        candidate_tokens = _team_tokens(_expand_city_aliases(candidate))
        if not candidate_tokens:
            continue
        overlap = len(incoming_tokens & candidate_tokens) / max(len(incoming_tokens), len(candidate_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = candidate

    # Require a meaningful token overlap to avoid false matches.
    if best_overlap >= 0.5:
        return best_match
    return None


def _coerce_injuries(payload):
    """Normalize injury payloads to a {normalized_team_name: {player: status}} mapping."""
    normalized = {}
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = []
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            team = rec.get("team_name") or rec.get("team")
            player = rec.get("player") or rec.get("player_name") or rec.get("name")
            status = rec.get("status", "")
            iterable.append((team, {player: status}))
    else:
        return normalized

    for team, players in iterable:
        team_key = normalize_team_name(team)
        if not team_key:
            continue
        normalized.setdefault(team_key, {})
        if isinstance(players, dict):
            for player, status in players.items():
                if str(player or "").strip():
                    normalized[team_key][str(player).strip()] = str(status or "").strip().lower()
        elif isinstance(players, list):
            for rec in players:
                if not isinstance(rec, dict):
                    continue
                player = rec.get("player") or rec.get("player_name") or rec.get("name")
                status = rec.get("status", "")
                if str(player or "").strip():
                    normalized[team_key][str(player).strip()] = str(status or "").strip().lower()
    return normalized


def load_injuries():
    path = "sports_betting/data/injuries/injuries.json"
    fallback_path = "sports_betting/data/inputs/injuries.json"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            injuries = json.load(f)
    elif os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            injuries = json.load(f)
    else:
        print("No injury data available — using empty dict")
        injuries = {}
    injuries = _coerce_injuries(injuries)

    print("[INJURY STATUS]")
    print(f"Teams with injuries: {len(injuries)}")
    for team, players in list(injuries.items())[:5]:
        print(team, list(players.keys())[:3])

    return injuries


STAR_PLAYERS = {
    "stephen curry",
    "lebron james",
    "kevin durant",
    "nikola jokic",
    "luka doncic",
    "giannis antetokounmpo",
    "jayson tatum",
    "joel embiid",
    "shai gilgeous alexander",
    "anthony davis",
}


def calculate_injury_impact(player_status_map):
    """Weighted impact score; stars are weighted more heavily than role players."""
    impact = 0.0
    for player, status in player_status_map.items():
        normalized_status = str(status or "").strip().lower()
        status_weight = 0.0
        if normalized_status in {"out", "inactive", "ir"}:
            status_weight = 1.0
        elif normalized_status == "doubtful":
            status_weight = 0.75
        elif normalized_status in {"questionable", "day-to-day", "day to day"}:
            status_weight = 0.5
        elif normalized_status == "probable":
            status_weight = 0.2
        if status_weight <= 0:
            continue

        is_star = normalize_player_name(player) in STAR_PLAYERS
        player_weight = 2.0 if is_star else 1.0
        impact += status_weight * player_weight
    return impact


def match_injury_impact(predictions_df, injuries):
    """Map injuries onto home/away teams and compute side-specific + combined impact."""
    out = predictions_df.copy()
    out["injury_impact_home"] = 0.0
    out["injury_impact_away"] = 0.0
    for idx, row in out.iterrows():
        home_key = resolve_injury_team_key(row.get("home_team"), injuries)
        away_key = resolve_injury_team_key(row.get("away_team"), injuries)
        if home_key:
            out.at[idx, "injury_impact_home"] = calculate_injury_impact(injuries.get(home_key, {}))
        if away_key:
            out.at[idx, "injury_impact_away"] = calculate_injury_impact(injuries.get(away_key, {}))
    out["combined_injury_impact"] = out["injury_impact_home"] + out["injury_impact_away"]
    return out


def get_team_injury_penalty(team_name, injuries):
    resolved_team = resolve_injury_team_key(team_name, injuries)
    if not resolved_team:
        return 0.0
    # Convert impact points into a probability penalty.
    return 0.02 * calculate_injury_impact(injuries.get(resolved_team, {}))
