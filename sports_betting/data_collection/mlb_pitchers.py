"""MLB probable pitcher fetching and lightweight ERA mapping."""

from __future__ import annotations

import requests

SCOREBOARD_API_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

# Temporary fallback values until pitcher stats API integration is added.
PITCHER_ERA: dict[str, float] = {
    "Gerrit Cole": 3.10,
    "Nick Pivetta": 4.20,
    "Zack Wheeler": 2.95,
    "Aaron Nola": 3.70,
    "Corbin Burnes": 3.15,
    "Pablo Lopez": 3.65,
    "Spencer Strider": 3.40,
    "Framber Valdez": 3.45,
    "Yoshinobu Yamamoto": 3.25,
    "Logan Webb": 3.30,
}


def get_probable_pitchers() -> dict[str, str]:
    """Fetch probable pitchers from ESPN's MLB scoreboard API as team->pitcher."""
    try:
        response = requests.get(SCOREBOARD_API_URL, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        pitchers: dict[str, str] = {}
        print("🚨 ESPN API PITCHERS:", pitchers)
        print("🚨 COUNT:", len(pitchers))
        return pitchers

    data = response.json()
    pitchers: dict[str, str] = {}

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]

        for competitor in comp.get("competitors", []):
            team = competitor.get("team", {})
            team_name = team.get("displayName")
            if not team_name:
                continue

            probables = competitor.get("probables", [])
            if probables:
                athlete = probables[0].get("athlete", {})
                pitcher_name = athlete.get("displayName")
                if pitcher_name:
                    pitchers[team_name.lower()] = pitcher_name

    print("🚨 ESPN API PITCHERS:", pitchers)
    print("🚨 COUNT:", len(pitchers))
    return pitchers


def fetch_mlb_probable_pitchers() -> dict[str, str]:
    """Backward-compatible wrapper used by the existing MLB pipeline."""
    return get_probable_pitchers()
