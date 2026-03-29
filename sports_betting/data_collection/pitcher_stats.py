"""Dynamic MLB pitcher statistics lookups."""

from __future__ import annotations

import requests


def get_pitcher_era(pitcher_name: str) -> float | None:
    """Fetch a pitcher's current season ERA from the MLB Stats API."""
    if not pitcher_name:
        return None

    try:
        search_url = "https://statsapi.mlb.com/api/v1/people/search"
        search_res = requests.get(search_url, params={"names": pitcher_name}, timeout=20).json()

        people = search_res.get("people", [])
        if not people:
            return None

        player_id = people[0]["id"]
        stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        stats_res = requests.get(
            stats_url,
            params={"stats": "season", "group": "pitching"},
            timeout=20,
        ).json()

        stats = stats_res.get("stats", [])
        if not stats:
            return None

        splits = stats[0].get("splits", [])
        if not splits:
            return None

        era = splits[0].get("stat", {}).get("era")
        return float(era) if era else None
    except Exception:
        return None


def build_pitcher_era_map(pitchers_dict: dict[str, str]) -> dict[str, float]:
    """Build a pitcher-name -> ERA mapping for all probable pitchers in scope."""
    era_map: dict[str, float] = {}

    for pitcher in {name for name in pitchers_dict.values() if name}:
        era = get_pitcher_era(pitcher)
        if era is not None:
            era_map[pitcher] = era

    return era_map
