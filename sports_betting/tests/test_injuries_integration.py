import json
from pathlib import Path

import pandas as pd

from sports_betting.data.fetch_injuries import compute_injury_impact
from sports_betting.data.load_injuries import load_injuries


def test_compute_injury_impact_uses_team_name_column():
    games = pd.DataFrame([{"home_team": "New York Rangers", "away_team": "Boston Bruins"}])
    injuries = pd.DataFrame(
        [
            {"sport": "nhl", "team_name": "New York Rangers", "player": "Player A", "status": "out"},
            {"sport": "nhl", "team_name": "Boston Bruins", "player": "Player B", "status": "questionable"},
        ]
    )

    out = compute_injury_impact(games, injuries)
    assert out.loc[0, "injury_impact_home"] > 0
    assert out.loc[0, "injury_impact_away"] > 0
    assert "injury_impact_diff" in out.columns


def test_compute_injury_impact_keeps_original_team_names():
    games = pd.DataFrame([{"home_team": "New York Rangers", "away_team": "Boston Bruins"}])
    injuries = pd.DataFrame(
        [{"sport": "nhl", "team_name": "New York Rangers", "player": "Player A", "status": "out"}]
    )

    out = compute_injury_impact(games, injuries)

    assert out.loc[0, "home_team"] == "New York Rangers"
    assert out.loc[0, "away_team"] == "Boston Bruins"


def test_compute_injury_impact_boosts_star_player_weight():
    games = pd.DataFrame([{"home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"}])
    injuries = pd.DataFrame(
        [
            {"sport": "nba", "team_name": "Los Angeles Lakers", "player": "LeBron James", "status": "out"},
            {"sport": "nba", "team_name": "Boston Celtics", "player": "Depth Player", "status": "out"},
        ]
    )

    out = compute_injury_impact(games, injuries)
    assert out.loc[0, "injury_impact_home"] > out.loc[0, "injury_impact_away"]


def test_load_injuries_normalizes_team_keys(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    injury_dir = root / "sports_betting" / "data" / "injuries"
    injury_dir.mkdir(parents=True)
    payload = [
        {"team_name": "Los Angeles Lakers", "player_name": "LeBron James", "status": "Out"},
        {"team_name": "Toronto Maple Leafs", "player_name": "Player X", "status": "Questionable"},
    ]
    (injury_dir / "injuries.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.chdir(root)
    injuries = load_injuries()
    assert "los angeles lakers" in injuries
    assert injuries["los angeles lakers"]["LeBron James"] == "out"
    assert "toronto maple leafs" in injuries
