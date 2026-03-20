"""Build historical NBA game-level features from Basketball Reference."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://www.basketball-reference.com"
DEFAULT_OUTPUT = Path("sports_betting/data/historical/nba_historical.csv")


@dataclass
class EloConfig:
    start_rating: float = 1500.0
    k_factor: float = 20.0


def _fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def _month_urls_for_season(season_end_year: int) -> list[str]:
    season_url = f"{BASE_URL}/leagues/NBA_{season_end_year}_games.html"
    html = _fetch_html(season_url)
    hrefs = sorted(
        set(re.findall(r'href="(/leagues/NBA_\d+_games-[a-z]+\.html)"', html))
    )
    if not hrefs:
        raise RuntimeError(f"No monthly schedule links found for season {season_end_year}.")
    return [f"{BASE_URL}{href}" for href in hrefs]


def _read_schedule_table(url: str) -> pd.DataFrame:
    html = _fetch_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    df = tables[0].copy()
    if "Date" not in df.columns:
        return pd.DataFrame()

    keep_cols = {
        "Date": "event_date",
        "Visitor/Neutral": "away_team",
        "Home/Neutral": "home_team",
        "PTS": "away_score",
        "PTS.1": "home_score",
    }
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[list(keep_cols)].rename(columns=keep_cols)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")

    # Drop postponed or malformed rows.
    df = df.dropna(subset=["event_date", "away_team", "home_team", "away_score", "home_score"])
    df["away_score"] = df["away_score"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    return df


def _fetch_regular_season_games(seasons: Iterable[int]) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    for season_end_year in seasons:
        for month_url in _month_urls_for_season(season_end_year):
            month_df = _read_schedule_table(month_url)
            if month_df.empty:
                continue
            month_df["season"] = season_end_year
            all_frames.append(month_df)

    if not all_frames:
        raise RuntimeError("No NBA schedule data fetched.")

    games = pd.concat(all_frames, ignore_index=True)
    games = games.sort_values(["event_date", "home_team", "away_team"]).reset_index(drop=True)

    # Deduplicate by date/home/away in case monthly pages overlap.
    games = games.drop_duplicates(subset=["event_date", "home_team", "away_team"], keep="first")
    games["game_id"] = (
        games["event_date"].dt.strftime("%Y%m%d")
        + "_"
        + games["away_team"].str.replace(" ", "", regex=False)
        + "_at_"
        + games["home_team"].str.replace(" ", "", regex=False)
    )
    return games


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _build_features(games: pd.DataFrame, elo_config: EloConfig) -> pd.DataFrame:
    team_elo: dict[str, float] = defaultdict(lambda: elo_config.start_rating)
    last_game_date: dict[str, pd.Timestamp] = {}
    point_diff_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))

    rows: list[dict[str, object]] = []

    for game in games.itertuples(index=False):
        home_team = game.home_team
        away_team = game.away_team
        event_date = pd.Timestamp(game.event_date)

        elo_home = float(team_elo[home_team])
        elo_away = float(team_elo[away_team])

        if home_team in last_game_date:
            rest_days_home = (event_date - last_game_date[home_team]).days
        else:
            rest_days_home = np.nan

        if away_team in last_game_date:
            rest_days_away = (event_date - last_game_date[away_team]).days
        else:
            rest_days_away = np.nan

        last5_home = float(np.mean(point_diff_history[home_team])) if point_diff_history[home_team] else 0.0
        last5_away = float(np.mean(point_diff_history[away_team])) if point_diff_history[away_team] else 0.0

        home_score = int(game.home_score)
        away_score = int(game.away_score)
        home_win = int(home_score > away_score)

        rows.append(
            {
                "game_id": game.game_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "elo_home": elo_home,
                "elo_away": elo_away,
                "elo_diff": elo_home - elo_away,
                "rest_days_home": rest_days_home,
                "rest_days_away": rest_days_away,
                "rest_diff": (
                    (rest_days_home if pd.notna(rest_days_home) else 0)
                    - (rest_days_away if pd.notna(rest_days_away) else 0)
                ),
                "last5_net_rating_home": last5_home,
                "last5_net_rating_away": last5_away,
                "last5_net_rating_diff": last5_home - last5_away,
                "net_rating_home": last5_home,
                "net_rating_away": last5_away,
                "net_rating_diff": last5_home - last5_away,
                "home_moneyline": np.nan,
                "spread": np.nan,
            }
        )

        expected_home = _expected_score(elo_home, elo_away)
        expected_away = 1.0 - expected_home
        actual_home = float(home_win)
        actual_away = 1.0 - actual_home

        team_elo[home_team] = elo_home + elo_config.k_factor * (actual_home - expected_home)
        team_elo[away_team] = elo_away + elo_config.k_factor * (actual_away - expected_away)

        point_diff_history[home_team].append(home_score - away_score)
        point_diff_history[away_team].append(away_score - home_score)
        last_game_date[home_team] = event_date
        last_game_date[away_team] = event_date

    output = pd.DataFrame(rows)
    output = output.sort_values(["event_date", "game_id"]).reset_index(drop=True)
    return output


def _default_end_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year - 1


def build_nba_historical_csv(output_path: Path = DEFAULT_OUTPUT, num_seasons: int = 3, end_season: int | None = None) -> Path:
    if num_seasons < 1:
        raise ValueError("num_seasons must be >= 1")

    end_season = end_season or _default_end_season()
    seasons = list(range(end_season - num_seasons + 1, end_season + 1))

    games = _fetch_regular_season_games(seasons)
    dataset = _build_features(games, EloConfig())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NBA historical CSV from Basketball Reference.")
    parser.add_argument("--num-seasons", type=int, default=3, help="Number of completed seasons to include.")
    parser.add_argument("--end-season", type=int, default=None, help="Season end year (e.g., 2025 for 2024-25).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    args = parser.parse_args()

    out_path = build_nba_historical_csv(
        output_path=args.output,
        num_seasons=args.num_seasons,
        end_season=args.end_season,
    )
    print(f"Wrote NBA historical dataset to {out_path}")


if __name__ == "__main__":
    main()
