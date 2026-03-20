import json
import pandas as pd
from urllib.request import urlopen


def _normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
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
    out = df_games.copy()
    OUT_STATUSES = ["Out", "Out for season"]
    QUESTIONABLE_STATUSES = ["Questionable", "Doubtful"]

    injuries = df_injuries.copy()
    if injuries.empty:
        injuries = pd.DataFrame(columns=["team", "player", "status"])
    if "team" not in injuries.columns:
        injuries["team"] = ""
    if "status" not in injuries.columns:
        injuries["status"] = ""

    injuries["team"] = injuries["team"].apply(_normalize_team_name)
    injuries["status"] = injuries["status"].astype(str)
    out["home_team"] = out.get("home_team", "").astype(str)
    out["away_team"] = out.get("away_team", "").astype(str)
    out["_home_team_norm"] = out["home_team"].apply(_normalize_team_name)
    out["_away_team_norm"] = out["away_team"].apply(_normalize_team_name)

    def get_team_impact(team: str) -> float:
        team_injuries = injuries[injuries["team"] == team]

        out_count = team_injuries[team_injuries["status"].isin(OUT_STATUSES)].shape[0]
        questionable_count = team_injuries[team_injuries["status"].isin(QUESTIONABLE_STATUSES)].shape[0]

        return (out_count * 1.0) + (questionable_count * 0.5)

    out["injury_impact_home"] = out["_home_team_norm"].apply(get_team_impact)
    out["injury_impact_away"] = out["_away_team_norm"].apply(get_team_impact)

    out["injury_impact_diff"] = out["injury_impact_home"] - out["injury_impact_away"]
    out = out.drop(columns=["_home_team_norm", "_away_team_norm"])

    return out
