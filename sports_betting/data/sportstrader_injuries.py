import os

import requests


def fetch_sportstrader_injuries():
    api_key = os.getenv("SPORTTRADER_API_KEY")

    if not api_key:
        print("SPORTTRADER_API_KEY not found")
        return {}

    url = "https://api.sportstrader.com/v1/injuries/nba"  # adjust if needed

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
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
    for item in data.get("injuries", []):
        team = item.get("team", "").lower()
        player = item.get("player_name", "")
        status = item.get("status", "").lower()

        if not team or not player:
            continue

        if team not in injuries:
            injuries[team] = {}

        injuries[team][player] = status

    print(f"[SPORTSTRADER] Loaded injuries for {len(injuries)} teams")

    return injuries
