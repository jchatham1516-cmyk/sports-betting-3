import json
import os


def load_injuries():
    path = "data/inputs/injuries.json"

    if not os.path.exists(path):
        print("No injuries.json found — skipping injury adjustments")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_team_injury_penalty(team_name, injuries):
    team_name = team_name.lower()

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
        status = status.lower()

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
