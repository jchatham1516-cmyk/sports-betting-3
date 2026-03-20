import json

import requests
from bs4 import BeautifulSoup

URL = "https://www.espn.com/nba/injuries"
OUTPUT_PATH = "sports_betting/data/injuries/injuries.json"


def _safe_team_name(table) -> str:
    # ESPN wraps each team table in a section where the heading contains team name.
    section = table.find_parent("section")
    if section:
        heading = section.find(["h2", "h3"])
        if heading:
            return heading.get_text(" ", strip=True).lower()
    # Fallback to nearest label/span in case the page structure changes.
    label = table.find_previous(["h2", "h3", "span"])
    if label:
        return label.get_text(" ", strip=True).lower()
    return "unknown"


def fetch_espn_injuries():
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    injuries = {}
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")[1:]
        team = _safe_team_name(table)
        if team not in injuries:
            injuries[team] = {}

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            player = cols[0].text.strip()
            status = cols[2].text.strip().lower()

            if player:
                injuries[team][player] = status

    # Remove empty team buckets that come from non-team tables.
    return {team: players for team, players in injuries.items() if players}


def save_injuries():
    injuries = fetch_espn_injuries()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(injuries, f, indent=2)

    print(f"[ESPN] Saved injuries for {len(injuries)} teams")


if __name__ == "__main__":
    save_injuries()
