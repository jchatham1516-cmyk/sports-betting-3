import json
import os


def normalize_team_name(name):
    return str(name).lower().strip()


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


def get_team_injury_penalty(team_name, injuries):
    team_name = normalize_team_name(team_name)

    if team_name not in injuries:
        return 0.0

    penalty = 0.0

    STAR_PLAYERS = [
        "Stephen Curry",
        "LeBron James",
        "Kevin Durant",
        "Nikola Jokic",
        "Luka Doncic",
        "Giannis Antetokounmpo",
    ]

    for player, status in injuries[team_name].items():
        status = str(status).lower()

        if player in STAR_PLAYERS:
            if status == "out":
                penalty += 0.10  # HUGE impact
            elif status == "questionable":
                penalty += 0.05
        else:
            if status == "out":
                penalty += 0.03
            elif status == "questionable":
                penalty += 0.015

    return penalty
