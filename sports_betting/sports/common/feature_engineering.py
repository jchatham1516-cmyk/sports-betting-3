"""Feature engineering for injuries, travel fatigue, market and efficiency context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sports_betting.sports.common.injuries import load_injury_frame, normalize_status, summarize_team_injuries

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


def add_injury_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    injuries = load_injury_frame(sport, data_root)
    team_summary = summarize_team_injuries(injuries, sport)
    out = games_df.copy()

    # Persist a team summary artifact for debugging and reporting.
    external = data_root / "external"
    external.mkdir(parents=True, exist_ok=True)
    team_summary.to_csv(external / f"{sport}_injury_team_summary.csv", index=False)

    default_cols = [
        "injury_impact_home",
        "injury_impact_away",
        "injury_impact_diff",
        "starter_out_count_home",
        "starter_out_count_away",
        "star_player_out_home",
        "star_player_out_away",
        "star_player_out_flag_home",
        "star_player_out_flag_away",
        "qb_out_flag_home",
        "qb_out_flag_away",
        "starting_goalie_out_flag_home",
        "starting_goalie_out_flag_away",
        "offensive_injury_weight_home",
        "offensive_injury_weight_away",
        "offensive_injury_weight_diff",
        "defensive_injury_weight_home",
        "defensive_injury_weight_away",
        "defensive_injury_weight_diff",
        "injury_confidence_score_home",
        "injury_confidence_score_away",
        "injury_confidence_score",
        "injury_data_stale_flag",
    ]

    if team_summary.empty:
        for col in default_cols:
            out[col] = 0.0
        return out

    home = team_summary.add_suffix("_home").rename(columns={"team_home": "home_team"})
    away = team_summary.add_suffix("_away").rename(columns={"team_away": "away_team"})
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left").fillna(0.0)

    out["injury_impact_diff"] = out["injury_impact_away"] - out["injury_impact_home"]
    out["offensive_injury_weight_diff"] = out.get("offensive_injury_weight_away", 0.0) - out.get("offensive_injury_weight_home", 0.0)
    out["defensive_injury_weight_diff"] = out.get("defensive_injury_weight_away", 0.0) - out.get("defensive_injury_weight_home", 0.0)
    out["star_player_out_home"] = out.get("star_player_out_home", 0.0)
    out["star_player_out_away"] = out.get("star_player_out_away", 0.0)
    out["star_player_out_flag_home"] = out["star_player_out_home"]
    out["star_player_out_flag_away"] = out["star_player_out_away"]
    out["injury_confidence_score"] = 0.5 * (
        pd.to_numeric(out.get("injury_confidence_score_home", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("injury_confidence_score_away", 0.0), errors="coerce").fillna(0.0)
    )
    out["injury_data_stale_flag"] = (
        pd.to_numeric(out.get("injury_data_stale_home", 0.0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("injury_data_stale_away", 0.0), errors="coerce").fillna(0)
        > 0
    ).astype(float)
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
            road_trip = 0 if is_home and state.last_was_home is not False else state.consecutive_road_games + (0 if is_home else 1)

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
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left").fillna(0.0)
    for f in feature_set:
        out[f"{f}_diff"] = out[f"{f}_away"] - out[f"{f}_home"]

    out["offensive_rating_diff"] = out.get("offensive_rating_diff", out.get("epa_per_play_diff", out.get("xgf_diff", 0.0)))
    out["defensive_rating_diff"] = out.get("defensive_rating_diff", out.get("defensive_efficiency_diff", out.get("xga_diff", 0.0)))
    out["net_rating_diff"] = out.get("net_rating_diff", out.get("success_rate_diff", out.get("xgf_pct_diff", 0.0)))
    out["pace"] = out.get("pace_home", 0.0) + out.get("pace_away", 0.0)
    return out


def add_market_context_features(games_df: pd.DataFrame) -> pd.DataFrame:
    out = games_df.copy()

    def _series(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    out["market_prob"] = _series("market_prob")
    out["model_prob"] = _series("model_prob")
    out["edge"] = _series("edge")
    out["expected_value"] = _series("expected_value")
    out["open_line"] = _series("open_line") if "open_line" in out.columns else _series("spread_line")
    out["current_line"] = _series("current_line") if "current_line" in out.columns else _series("spread_line")
    out["line_movement"] = out["current_line"] - out["open_line"]
    out["public_favorite_bias_flag"] = (_series("home_odds") < -140).astype(float)
    out["favorite_inflation_flag"] = (out["line_movement"].abs() >= 1.5).astype(float)
    out["underdog_inflation_flag"] = (out["line_movement"] <= -1.5).astype(float)
    out["clv_placeholder"] = _series("clv_placeholder")
    return out


def enrich_with_context_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    out = add_injury_features(games_df, sport, data_root)
    out = add_travel_fatigue_features(out, sport, data_root)
    out = add_efficiency_features(out, sport, data_root)
    out = add_market_context_features(out)
    out["injury_impact"] = out.get("injury_impact_diff", 0.0)
    out["home_indicator"] = pd.to_numeric(out["home_indicator"], errors="coerce").fillna(1.0) if "home_indicator" in out.columns else 1.0
    return out.fillna(0.0)
