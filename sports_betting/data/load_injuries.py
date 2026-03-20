import json
import os

from sports_betting.data.sportstrader_injuries import fetch_sportstrader_injuries


def normalize_team_name(name):
    return str(name).lower().strip()


def load_injuries():
    # Try API first
    api_data = fetch_sportstrader_injuries()

    if api_data:
        injuries = api_data
    else:
        # Fallback to manual file
        path = "sports_betting/data/inputs/injuries.json"

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                injuries = json.load(f)
        else:
            print("No injury data available — using empty dict")
            injuries = {}

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
