import os

import requests


def fetch_sportstrader_injuries():
    api_key = os.getenv("SPORTRADAR_API_KEY")

    if not api_key:
        print("SPORTRADAR_API_KEY not found")
        return {}
    else:
        print("SPORTRADAR_API_KEY loaded successfully")

    url = "https://api.sportradar.com/nba/trial/v8/en/league/injuries.json"  # adjust if needed

    headers = {
        "x-api-key": api_key,
    }

    try:
        print(f"[SPORTSRADAR] Using API key: {api_key[:5]}***")
        print(f"[SPORTSRADAR] Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"SPORTSTRADER API error: {response.status_code}")
            return {}

        data = response.json()

    except Exception as e:
        print(f"SPORTSTRADER request failed: {e}")
        return {}

    injuries = {}

    # Normalize response → {team: {player: status}}
    for team in data.get("teams", []):
        team_name = f"{team.get('market', '')} {team.get('name', '')}".lower()

        for player in team.get("players", []):
            if "injury" not in player:
                continue

            player_name = player.get("full_name")
            status = player["injury"].get("status", "").lower()

            if team_name not in injuries:
                injuries[team_name] = {}

            injuries[team_name][player_name] = status

    print(f"[SPORTSRADAR] Loaded injuries for {len(injuries)} teams")

    return injuries
