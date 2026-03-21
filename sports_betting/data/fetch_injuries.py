import json
import pandas as pd
from urllib.request import urlopen


def _normalize_team_name(name: str | None) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(name)).split())


def fetch_nba_injuries() -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    with urlopen(url, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    injuries = []

    for team in data.get("teams", []):
        team_name = team.get("team", {}).get("displayName")

        for athlete in team.get("injuries", []):
            player_name = athlete.get("athlete", {}).get("displayName")
            status = athlete.get("status")

            injuries.append(
                {
                    "team": _normalize_team_name(team_name),
                    "player": player_name,
                    "status": status,
                }
            )

    df = pd.DataFrame(injuries)
    if df.empty:
        return pd.DataFrame(columns=["team", "player", "status"])
    return df


def compute_injury_impact(df_games: pd.DataFrame, df_injuries: pd.DataFrame) -> pd.DataFrame:
    """Refined injury impact calculation for teams based on player injuries."""
    out = df_games.copy()
    injuries = df_injuries.copy()

    if "home_team" not in out.columns:
        out["home_team"] = ""
    if "away_team" not in out.columns:
        out["away_team"] = ""

    if injuries.empty:
        injuries = pd.DataFrame(columns=["team", "player", "status"])
    if "team" not in injuries.columns:
        injuries["team"] = ""
    if "player" not in injuries.columns:
        injuries["player"] = ""

    # Normalize team names
    out["home_team"] = out["home_team"].astype(str).apply(_normalize_team_name)
    out["away_team"] = out["away_team"].astype(str).apply(_normalize_team_name)
    injuries["team"] = injuries["team"].astype(str).apply(_normalize_team_name)

    # Mapping injury data to teams
    injury_map = injuries.groupby("team")["player"].apply(len)

    # Apply injury impact mapping to teams
    out["injury_impact_home"] = out["home_team"].map(injury_map).fillna(0)
    out["injury_impact_away"] = out["away_team"].map(injury_map).fillna(0)

    # Calculate the difference between away team and home team injury impact
    out["injury_impact_diff"] = out["injury_impact_away"] - out["injury_impact_home"]

    return out
