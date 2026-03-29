"""ESPN MLB probable pitcher scraper adapter."""

from __future__ import annotations

import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

REQUIRED_COLUMNS = ["home_team_norm", "away_team_norm", "home_pitcher", "away_pitcher"]


def _normalize_team_name(value: object) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(value or "").strip().lower())
    return " ".join(cleaned.split())


def _placeholder_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "home_team_norm": "__unknown_home__",
                "away_team_norm": "__unknown_away__",
                "home_pitcher": None,
                "away_pitcher": None,
            }
        ],
        columns=REQUIRED_COLUMNS,
    )


def fetch_mlb_probable_pitchers_espn() -> pd.DataFrame:
    """Best-effort ESPN probable pitchers fetch that always returns rows."""
    print("\n[ESPN SCRAPER STARTED]")
    try:
        response = requests.get(
            "https://www.espn.com/mlb/scoreboard",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[MLB ESPN WARNING] ESPN request failed: {exc}")
        return _placeholder_df()

    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, object]] = []

    for card in soup.find_all("section"):
        section_text = card.get_text(" ", strip=True)
        teams = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", section_text)
        if len(teams) < 2:
            continue
        probable = re.findall(r"Probable(?:\s+Pitchers?)?\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z\-']+)+)", section_text, flags=re.IGNORECASE)
        away_pitcher = probable[0] if len(probable) > 0 else None
        home_pitcher = probable[1] if len(probable) > 1 else None
        rows.append(
            {
                "home_team_norm": _normalize_team_name(teams[1]),
                "away_team_norm": _normalize_team_name(teams[0]),
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
            }
        )

    if not rows:
        for script in soup.find_all("script"):
            payload_text = script.string or script.get_text("", strip=True)
            if not payload_text or "probable" not in payload_text.lower() or "pitch" not in payload_text.lower():
                continue
            for candidate in re.finditer(r"\{.*\}", payload_text):
                try:
                    decoded = json.loads(candidate.group(0))
                except json.JSONDecodeError:
                    continue
                if not isinstance(decoded, dict):
                    continue
                home_team = decoded.get("homeTeam") or decoded.get("home_team")
                away_team = decoded.get("awayTeam") or decoded.get("away_team")
                home_pitcher = decoded.get("homeProbablePitcher") or decoded.get("home_pitcher")
                away_pitcher = decoded.get("awayProbablePitcher") or decoded.get("away_pitcher")
                if home_team and away_team:
                    rows.append(
                        {
                            "home_team_norm": _normalize_team_name(home_team),
                            "away_team_norm": _normalize_team_name(away_team),
                            "home_pitcher": home_pitcher,
                            "away_pitcher": away_pitcher,
                        }
                    )
            if rows:
                break

    out = pd.DataFrame(rows, columns=REQUIRED_COLUMNS) if rows else _placeholder_df()
    out = out.drop_duplicates(subset=["home_team_norm", "away_team_norm"], keep="last")
    if out.empty:
        out = _placeholder_df()

    print("\n[ESPN PITCHER DEBUG]")
    print(out.head(20))
    print("TOTAL GAMES SCRAPED:", len(out))
    return out
