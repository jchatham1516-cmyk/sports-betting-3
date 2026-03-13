"""Feature engineering for injuries, travel fatigue, and efficiency metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "day-to-day": "day-to-day",
    "day to day": "day-to-day",
    "ir": "ir",
    "inactive": "inactive",
}

STATUS_WEIGHT = {
    "out": 1.0,
    "ir": 1.0,
    "inactive": 1.0,
    "doubtful": 0.75,
    "questionable": 0.45,
    "day-to-day": 0.35,
    "probable": 0.15,
}

SPORT_EFFICIENCY_FEATURES: dict[str, list[str]] = {
    "nba": [
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "pace",
        "true_shooting",
        "effective_fg",
        "turnover_rate",
        "rebound_rate",
        "free_throw_rate",
        "top_3_player_impact_sum",
        "top_5_rotation_impact_sum",
        "on_off_impact_sum",
    ],
    "nfl": [
        "epa_per_play",
        "success_rate",
        "yards_per_play",
        "red_zone_efficiency",
        "pressure_rate",
        "sack_rate",
        "explosive_play_rate",
        "qb_efficiency_metric",
        "offensive_line_grade_proxy",
        "defensive_efficiency",
        "special_teams_impact",
    ],
    "nhl": [
        "xgf",
        "xga",
        "xgf_pct",
        "shot_share",
        "special_teams_efficiency",
        "goalie_save_strength",
        "goalie_xgsaved_proxy",
        "top_line_impact",
        "defensive_pair_impact",
    ],
}


@dataclass
class TeamTravelState:
    last_game_date: pd.Timestamp | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    last_tz: float = 0.0
    consecutive_road_games: int = 0
    last_was_home: bool | None = None


def normalize_status(status: str) -> str:
    s = str(status or "").strip().lower()
    return STATUS_MAP.get(s, "questionable")


def _to_bool(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def load_injury_reports(sport: str, data_root: Path) -> pd.DataFrame:
    path = data_root / "external" / f"{sport}_injuries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["team", "player", "status", "impact_rating", "is_starter", "is_star", "unit", "position", "starting_goalie", "qb"]:
        if col not in df.columns:
            df[col] = 0 if col in {"impact_rating", "is_starter", "is_star", "starting_goalie", "qb"} else ""
    df["status_norm"] = df["status"].map(normalize_status)
    df["status_weight"] = df["status_norm"].map(STATUS_WEIGHT).fillna(0.4)
    df["impact_rating"] = pd.to_numeric(df["impact_rating"], errors="coerce").fillna(0.8)
    df["weighted_impact"] = df["impact_rating"] * df["status_weight"]
    df["is_starter"] = df["is_starter"].map(_to_bool)
    df["is_star"] = df["is_star"].map(_to_bool)
    df["starting_goalie"] = df["starting_goalie"].map(_to_bool)
    df["qb"] = df["qb"].map(_to_bool)
    df["unit"] = df["unit"].astype(str).str.lower().replace({"": "offense"})
    return df


def _team_injury_summary(inj_df: pd.DataFrame, sport: str) -> pd.DataFrame:
    if inj_df.empty:
        return pd.DataFrame()
    grouped = inj_df.groupby("team", dropna=False)
    out = grouped.agg(
        injury_impact=("weighted_impact", "sum"),
        starter_out_count=("is_starter", lambda s: int(((s == 1) & (inj_df.loc[s.index, "status_weight"] >= 0.75)).sum())),
        star_player_out_flag=("is_star", lambda s: int(((s == 1) & (inj_df.loc[s.index, "status_weight"] >= 0.75)).any())),
        offensive_injury_weight=("weighted_impact", lambda s: float(s[inj_df.loc[s.index, "unit"] == "offense"].sum())),
        defensive_injury_weight=("weighted_impact", lambda s: float(s[inj_df.loc[s.index, "unit"] == "defense"].sum())),
        starting_goalie_out_flag=("starting_goalie", lambda s: int(((s == 1) & (inj_df.loc[s.index, "status_weight"] >= 0.75)).any())),
        qb_out_flag=("qb", lambda s: int(((s == 1) & (inj_df.loc[s.index, "status_weight"] >= 0.75)).any())),
    ).reset_index()
    if sport != "nhl":
        out["starting_goalie_out_flag"] = 0
    if sport != "nfl":
        out["qb_out_flag"] = 0
    return out


def add_injury_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    injuries = load_injury_reports(sport, data_root)
    team_summary = _team_injury_summary(injuries, sport)
    out = games_df.copy()
    if team_summary.empty:
        for col in [
            "injury_impact_home",
            "injury_impact_away",
            "injury_impact_diff",
            "starter_out_count_home",
            "starter_out_count_away",
            "star_player_out_flag_home",
            "star_player_out_flag_away",
            "starting_goalie_out_flag_home",
            "starting_goalie_out_flag_away",
            "qb_out_flag_home",
            "qb_out_flag_away",
            "offensive_injury_weight_diff",
            "defensive_injury_weight_diff",
        ]:
            out[col] = 0.0
        return out

    home = team_summary.add_suffix("_home").rename(columns={"team_home": "home_team"})
    away = team_summary.add_suffix("_away").rename(columns={"team_away": "away_team"})
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left")
    out = out.fillna(0.0)
    out["injury_impact_home"] = out["injury_impact_home"]
    out["injury_impact_away"] = out["injury_impact_away"]
    out["injury_impact_diff"] = out["injury_impact_away"] - out["injury_impact_home"]
    out["offensive_injury_weight_diff"] = out.get("offensive_injury_weight_away", 0.0) - out.get("offensive_injury_weight_home", 0.0)
    out["defensive_injury_weight_diff"] = out.get("defensive_injury_weight_away", 0.0) - out.get("defensive_injury_weight_home", 0.0)
    return out


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _load_team_locations(data_root: Path, sport: str) -> pd.DataFrame:
    path = data_root / "external" / f"{sport}_team_locations.csv"
    if path.exists():
        locs = pd.read_csv(path)
        locs.columns = [c.strip().lower() for c in locs.columns]
        return locs
    return pd.DataFrame(columns=["team", "lat", "lon", "timezone"])


def _team_loc(team: str, locs: pd.DataFrame) -> tuple[float, float, float]:
    if not locs.empty and team in set(locs["team"]):
        row = locs[locs["team"] == team].iloc[0]
        return float(row.get("lat", 39.5)), float(row.get("lon", -98.35)), float(row.get("timezone", -5))
    seed = abs(hash(team)) % 1000
    return 25 + (seed % 250) / 10.0, -124 + (seed % 580) / 10.0, -8 + (seed % 60) / 10.0


def add_travel_fatigue_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    out = games_df.copy()
    out["event_date"] = pd.to_datetime(out.get("event_date"), errors="coerce", utc=True)
    locs = _load_team_locations(data_root, sport)

    states: dict[str, TeamTravelState] = {}
    for idx in out.sort_values("event_date").index:
        row = out.loc[idx]
        for side in ["home", "away"]:
            team = row[f"{side}_team"]
            opponent_side = "away" if side == "home" else "home"
            lat, lon, tz = _team_loc(str(team), locs)
            date = row["event_date"]
            state = states.setdefault(str(team), TeamTravelState())
            rest_days = 7.0
            travel = 0.0
            tz_shift = 0.0
            if state.last_game_date is not None and pd.notna(date):
                rest_days = max(0.0, float((date - state.last_game_date).days))
                if state.last_lat is not None and state.last_lon is not None:
                    travel = _haversine_miles(state.last_lat, state.last_lon, lat, lon)
                tz_shift = abs(tz - state.last_tz)
            is_home = side == "home"
            if is_home:
                road_trip = 0 if state.last_was_home is not False else state.consecutive_road_games
            else:
                road_trip = state.consecutive_road_games + 1

            out.loc[idx, f"rest_days_{side}"] = rest_days
            out.loc[idx, f"back_to_back_{side}"] = int(rest_days <= 1)
            out.loc[idx, f"three_in_four_{side}"] = int(rest_days <= 2)
            out.loc[idx, f"road_trip_length_{side}"] = road_trip
            out.loc[idx, f"timezone_shift_{side}"] = tz_shift
            out.loc[idx, f"travel_distance_{side}"] = travel

            state.last_game_date = date
            state.last_lat, state.last_lon, state.last_tz = lat, lon, tz
            state.last_was_home = is_home
            state.consecutive_road_games = 0 if is_home else road_trip

        out.loc[idx, "rest_diff"] = out.loc[idx, "rest_days_home"] - out.loc[idx, "rest_days_away"]
        out.loc[idx, "travel_distance"] = out.loc[idx, "travel_distance_away"] - out.loc[idx, "travel_distance_home"]
        out.loc[idx, "travel_fatigue_diff"] = (
            0.002 * out.loc[idx, "travel_distance"]
            + 0.35 * (out.loc[idx, "back_to_back_away"] - out.loc[idx, "back_to_back_home"])
            + 0.2 * (out.loc[idx, "three_in_four_away"] - out.loc[idx, "three_in_four_home"])
            + 0.05 * (out.loc[idx, "timezone_shift_away"] - out.loc[idx, "timezone_shift_home"])
        )

    return out.fillna(0.0)


def load_efficiency_metrics(sport: str, data_root: Path) -> pd.DataFrame:
    path = data_root / "external" / f"{sport}_efficiency.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def add_efficiency_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    out = games_df.copy()
    metrics = load_efficiency_metrics(sport, data_root)
    feature_set = SPORT_EFFICIENCY_FEATURES[sport]

    if metrics.empty or "team" not in metrics.columns:
        for f in feature_set:
            out[f"{f}_home"] = 0.0
            out[f"{f}_away"] = 0.0
            out[f"{f}_diff"] = 0.0
        return out

    for f in feature_set:
        if f not in metrics.columns:
            metrics[f] = 0.0

    home = metrics[["team", *feature_set]].copy().add_suffix("_home").rename(columns={"team_home": "home_team"})
    away = metrics[["team", *feature_set]].copy().add_suffix("_away").rename(columns={"team_away": "away_team"})
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left")
    out = out.fillna(0.0)
    for f in feature_set:
        out[f"{f}_diff"] = out[f"{f}_away"] - out[f"{f}_home"]

    # Shared aliases expected by model interfaces.
    out["offensive_rating_diff"] = out.get("offensive_rating_diff", out.get("epa_per_play_diff", out.get("xgf_diff", 0.0)))
    out["defensive_rating_diff"] = out.get("defensive_rating_diff", out.get("defensive_efficiency_diff", out.get("xga_diff", 0.0)))
    out["net_rating_diff"] = out.get("net_rating_diff", out.get("success_rate_diff", out.get("xgf_pct_diff", 0.0)))
    out["pace"] = out.get("pace_home", 0.0) + out.get("pace_away", 0.0)
    return out


def enrich_with_context_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    out = add_injury_features(games_df, sport, data_root)
    out = add_travel_fatigue_features(out, sport, data_root)
    out = add_efficiency_features(out, sport, data_root)
    out["injury_impact"] = out.get("injury_impact_diff", 0.0)
    home_indicator = out["home_indicator"] if "home_indicator" in out.columns else pd.Series(1.0, index=out.index)
    out["home_indicator"] = pd.to_numeric(home_indicator, errors="coerce").fillna(1.0)
    return out.fillna(0.0)
