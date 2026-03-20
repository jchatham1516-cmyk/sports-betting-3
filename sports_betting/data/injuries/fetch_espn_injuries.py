import json

import requests
from bs4 import BeautifulSoup

URL = "https://www.espn.com/nba/injuries"
OUTPUT_PATH = "sports_betting/data/injuries/injuries.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
}


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
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    print("[ESPN] Page fetched successfully")
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
            if len(cols) < 3:
                continue

            player = cols[0].get_text(" ", strip=True)
            status = cols[2].get_text(" ", strip=True)

            if player:
                injuries[team][player] = status

    # Remove empty team buckets that come from non-team tables.
    injuries = {team: players for team, players in injuries.items() if players}
    print(f"[ESPN] Found {len(injuries)} teams with injuries")
    return injuries


def run_injury_pipeline():
    injuries = fetch_espn_injuries()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(injuries, f, indent=2)

    print(f"[ESPN] Saved injuries to {OUTPUT_PATH}")


def save_injuries():
    # Backward-compatible wrapper for existing callers.
    run_injury_pipeline()


if __name__ == "__main__":
    run_injury_pipeline()
