"""MLB probable pitcher scraping and lightweight ERA mapping."""

from __future__ import annotations

import json
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

SCOREBOARD_URL = "https://www.espn.com/mlb/scoreboard"

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


def _clean_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _clean_pitcher_name(value: object) -> str:
    name = _clean_text(value)
    name = re.sub(r"\s*\([^\)]*\)", "", name).strip()
    return name


def _team_candidates(display_name: str, short_name: str) -> list[str]:
    display = _clean_text(display_name)
    short = _clean_text(short_name)
    out = [display, short]
    parts = display.split()
    if len(parts) > 1:
        out.append(parts[-1])
    return [candidate for candidate in out if candidate]


def _extract_games_from_state(state: dict[str, Any]) -> list[dict[str, str]]:
    games: list[dict[str, str]] = []
    events = (state.get("page") or {}).get("content") or {}
    score_data = events.get("scoreboard") or {}
    for event in score_data.get("events", []):
        competitors = ((event.get("competitions") or [{}])[0]).get("competitors") or []
        if len(competitors) < 2:
            continue

        home_comp = next((comp for comp in competitors if str(comp.get("homeAway", "")).lower() == "home"), None)
        away_comp = next((comp for comp in competitors if str(comp.get("homeAway", "")).lower() == "away"), None)
        if not home_comp or not away_comp:
            continue

        home_team = home_comp.get("team") or {}
        away_team = away_comp.get("team") or {}

        home_probable = home_comp.get("probableAthlete") or {}
        away_probable = away_comp.get("probableAthlete") or {}

        home_pitcher = _clean_pitcher_name(home_probable.get("displayName") or home_probable.get("fullName"))
        away_pitcher = _clean_pitcher_name(away_probable.get("displayName") or away_probable.get("fullName"))

        for key in _team_candidates(home_team.get("displayName", ""), home_team.get("shortDisplayName", "")):
            if home_pitcher:
                games.append({"team": key, "pitcher": home_pitcher})
        for key in _team_candidates(away_team.get("displayName", ""), away_team.get("shortDisplayName", "")):
            if away_pitcher:
                games.append({"team": key, "pitcher": away_pitcher})

    return games


def fetch_mlb_probable_pitchers() -> dict[str, str]:
    """Fetch probable pitchers from ESPN scoreboard as a team->pitcher mapping."""
    try:
        response = requests.get(
            SCOREBOARD_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    rows: list[dict[str, str]] = []
    for script in soup.find_all("script"):
        content = script.string or script.get_text("", strip=True)
        if not content or "__espnfitt__" not in content:
            continue
        match = re.search(r"window\.__espnfitt__\s*=\s*(\{.*\});?", content, flags=re.DOTALL)
        if not match:
            continue
        try:
            state = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        rows = _extract_games_from_state(state)
        if rows:
            break

    pitchers: dict[str, str] = {}
    for row in rows:
        team = row.get("team", "")
        pitcher = row.get("pitcher", "")
        if team and pitcher:
            pitchers[team] = pitcher

    return pitchers
