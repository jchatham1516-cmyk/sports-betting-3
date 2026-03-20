import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime

print("Fetching NBA historical data...")

# Pull multiple seasons
seasons = ["2023-24", "2022-23", "2021-22"]

all_games = []

for season in seasons:
    print(f"Loading season {season}...")
    games = leaguegamefinder.LeagueGameFinder(season_nullable=season).get_data_frames()[0]
    all_games.append(games)

df = pd.concat(all_games)

# Keep only relevant columns
df = df[[
    "GAME_DATE",
    "TEAM_NAME",
    "MATCHUP",
    "WL",
    "PTS"
]]

# Identify home/away
df["is_home"] = df["MATCHUP"].str.contains("vs.")

# Split into home/away pairs
games = []

grouped = df.groupby("GAME_DATE")

for date, group in grouped:
    if len(group) != 2:
        continue

    home = group[group["is_home"] == True]
    away = group[group["is_home"] == False]

    if len(home) == 1 and len(away) == 1:
        home_team = home.iloc[0]
        away_team = away.iloc[0]

        games.append({
            "event_date": date,
            "home_team": home_team["TEAM_NAME"],
            "away_team": away_team["TEAM_NAME"],
            "home_score": home_team["PTS"],
            "away_score": away_team["PTS"],
            "home_win": 1 if home_team["WL"] == "W" else 0
        })

games_df = pd.DataFrame(games)

# Sort by date
games_df = games_df.sort_values("event_date")
if "home_moneyline" not in games_df.columns:
    games_df["home_moneyline"] = 0
if "spread" not in games_df.columns:
    games_df["spread"] = 0
if "implied_home_prob" not in games_df.columns:
    games_df["implied_home_prob"] = games_df["home_moneyline"].apply(
        lambda odds: ((-odds) / ((-odds) + 100)) if odds < 0 else (100 / (odds + 100))
    )
games_df["spread_abs"] = games_df["spread"].abs()
games_df["is_favorite"] = (games_df["home_moneyline"] < 0).astype(int)

print("Total games:", len(games_df))

# Save
output_path = "sports_betting/data/historical/nba_historical.csv"
games_df.to_csv(output_path, index=False)

print("Saved to:", output_path)
