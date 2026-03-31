"""Feature engineering for injuries, travel fatigue, market and efficiency context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sports_betting.sports.common.injuries import load_injury_frame, normalize_status, normalize_team_name, summarize_team_injuries

INJURY_EDGE_EV_SCALING_FACTOR = 2.0
INJURY_MODEL_PROB_WEIGHT = 0.03
EDGE_SCALING_FACTOR = 1.5

SPORT_EFFICIENCY_FEATURES: dict[str, list[str]] = {
    "nba": [
        "off_rating",
        "def_rating",
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
        "top_rotation_efficiency_sum",
        "on_off_impact_sum",
        "on_off_proxy",
    ],
    "nfl": [
        "epa_per_play",
        "success_rate",
        "yards_per_play",
        "qb_efficiency",
        "red_zone_efficiency",
        "pressure_rate",
        "sack_rate",
        "explosive_play_rate",
        "qb_efficiency_metric",
        "offensive_line_grade_proxy",
        "defensive_efficiency",
        "special_teams_impact",
        "weather_total_impact",
        "qb_impact",
    ],
    "nhl": [
        "xgf",
        "xga",
        "xgf%",
        "xgf_pct",
        "shot_share",
        "special_teams_efficiency",
        "goalie_save_strength",
        "goalie_xgsaved_proxy",
        "top_line_impact",
        "defensive_pair_impact",
        "goalie_strength",
        "special_teams_diff_proxy",
    ],
    "mlb": [
        "runs_per_game",
        "ops",
        "slugging",
        "team_k_rate",
        "pitcher_era",
        "pitcher_xera",
        "pitcher_whip",
        "pitcher_k_rate",
        "pitcher_last3_starts",
        "bullpen_era",
        "bullpen_usage",
    ],
    "soccer": [
        "xg_for",
        "xg_against",
        "form_last5",
        "home_advantage",
        "rest_days",
        "injury_impact",
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
    required_cols = ["home_team", "away_team"]
    if not all(col in games_df.columns for col in required_cols):
        print("[INJURY] Skipping injury features — missing team columns")
        return games_df

    injuries = load_injury_frame(sport, data_root)
    print("[INJURY STATUS]")
    print(f"Injuries rows: {len(injuries)}")
    team_summary = summarize_team_injuries(injuries, sport)
    out = games_df.copy()
    out["home_team"] = out["home_team"].apply(normalize_team_name)
    out["away_team"] = out["away_team"].apply(normalize_team_name)
    if not team_summary.empty:
        team_summary["team"] = team_summary["team"].apply(normalize_team_name)

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
        print("No injury data available — using zeros")
        for col in default_cols:
            out[col] = 0.0
        return out

    home = team_summary.add_suffix("_home").rename(columns={"team_home": "home_team"})
    away = team_summary.add_suffix("_away").rename(columns={"team_away": "away_team"})
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left").fillna(0.0)

    for col in ["injury_impact_home", "injury_impact_away"]:
        if col not in out.columns:
            out[col] = 0.0
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
    out["starter_out_count"] = out.get("starter_out_count_away", 0.0) - out.get("starter_out_count_home", 0.0)
    out["star_player_out_flag"] = out.get("star_player_out_flag_away", 0.0) - out.get("star_player_out_flag_home", 0.0)
    out["qb_out_flag"] = out.get("qb_out_flag_away", 0.0) - out.get("qb_out_flag_home", 0.0)
    out["starting_goalie_out_flag"] = out.get("starting_goalie_out_flag_away", 0.0) - out.get("starting_goalie_out_flag_home", 0.0)
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
    out["elo_diff"] = pd.to_numeric(out.get("elo_diff", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).astype(float)
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
            out.loc[idx, f"three_games_in_four_{side}"] = int(rest_days <= 2)
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
            + 0.2 * (out.loc[idx, "three_games_in_four_away"] - out.loc[idx, "three_games_in_four_home"])
            + 0.05 * (out.loc[idx, "timezone_shift_away"] - out.loc[idx, "timezone_shift_home"])
        )

    out["rest_days"] = out.get("rest_days_away", 0.0)
    out["back_to_back"] = out.get("back_to_back_away", 0.0)
    out["three_games_in_four"] = out.get("three_games_in_four_away", 0.0)
    out["time_zone_shift"] = out.get("timezone_shift_away", 0.0)
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

    for team_col in ("home_team", "away_team"):
        if team_col in out.columns:
            out[team_col] = out[team_col].apply(normalize_team_name)

    if metrics.empty or "team" not in metrics.columns:
        for f in feature_set:
            out[f"{f}_home"] = 0.0
            out[f"{f}_away"] = 0.0
            out[f"{f}_diff"] = 0.0
        return out

    for f in feature_set:
        if f not in metrics.columns:
            metrics[f] = 0.0

    metrics["team"] = metrics["team"].apply(normalize_team_name)

    home = metrics[["team", *feature_set]].copy().add_suffix("_home").rename(columns={"team_home": "home_team"})
    away = metrics[["team", *feature_set]].copy().add_suffix("_away").rename(columns={"team_away": "away_team"})
    out = out.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left").fillna(0.0)
    for f in feature_set:
        out[f"{f}_diff"] = out[f"{f}_away"] - out[f"{f}_home"]

    if sport == "nba":
        out["offensive_rating_diff"] = out.get("offensive_rating_diff", 0.0)
        out["defensive_rating_diff"] = out.get("defensive_rating_diff", 0.0)
        out["net_rating_diff"] = out.get("net_rating_diff", 0.0)
        out["pace"] = out.get("pace_home", 0.0) + out.get("pace_away", 0.0)
        out["off_rating_home"] = out.get("offensive_rating_home", 0.0)
        out["off_rating_away"] = out.get("offensive_rating_away", 0.0)
        out["def_rating_home"] = out.get("defensive_rating_home", 0.0)
        out["def_rating_away"] = out.get("defensive_rating_away", 0.0)
        out["pace_diff"] = out.get("pace_away", 0.0) - out.get("pace_home", 0.0)
        out["top_rotation_eff_diff"] = out.get("top_rotation_efficiency_sum_diff", out.get("top_5_rotation_impact_sum_diff", 0.0))
    elif sport == "nfl":
        out["qb_impact_home"] = out.get("qb_impact_home", out.get("qb_efficiency_metric_home", 0.0))
        out["qb_impact_away"] = out.get("qb_impact_away", out.get("qb_efficiency_metric_away", 0.0))
        out["qb_efficiency_home"] = out.get("qb_efficiency_home", out.get("qb_efficiency_metric_home", 0.0))
        out["qb_efficiency_away"] = out.get("qb_efficiency_away", out.get("qb_efficiency_metric_away", 0.0))
        out["qb_efficiency_diff"] = out.get("qb_efficiency_diff", out.get("qb_efficiency_metric_diff", 0.0))
    elif sport == "nhl":
        out["goalie_strength_home"] = out.get("goalie_strength_home", out.get("goalie_save_strength_home", 0.0))
        out["goalie_strength_away"] = out.get("goalie_strength_away", out.get("goalie_save_strength_away", 0.0))
        out["goalie_strength_diff"] = out.get("goalie_strength_away", 0.0) - out.get("goalie_strength_home", 0.0)
        out["special_teams_diff"] = out.get("special_teams_efficiency_diff", out.get("special_teams_impact_diff", 0.0))
        out["xgf_diff"] = out.get("xgf_diff", 0.0)
        out["xga_diff"] = out.get("xga_diff", 0.0)
        out["xGF_home"] = out.get("xgf_home", 0.0)
        out["xGF_away"] = out.get("xgf_away", 0.0)
        out["xGA_home"] = out.get("xga_home", 0.0)
        out["xGA_away"] = out.get("xga_away", 0.0)
        out["xGF%_home"] = out.get("xgf_pct_home", out.get("xgf%_home", 0.0))
        out["xGF%_away"] = out.get("xgf_pct_away", out.get("xgf%_away", 0.0))
        out["xGF%_diff"] = out.get("xgf_pct_diff", out.get("xgf%_diff", 0.0))
    return out



def add_elo_features(games_df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Compute preseason-initialized Elo values game-by-game with context adjustments."""
    out = games_df.copy()
    out["event_date"] = pd.to_datetime(out.get("event_date"), errors="coerce", utc=True)
    out["elo_diff"] = pd.to_numeric(out.get("elo_diff", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).astype(float)
    preseason = {"nba": 1505.0, "nfl": 1510.0, "nhl": 1500.0}.get(sport, 1500.0)
    hfa = {"nba": 65.0, "nfl": 45.0, "nhl": 35.0}.get(sport, 45.0)
    k = {"nba": 18.0, "nfl": 22.0, "nhl": 16.0}.get(sport, 18.0)
    ratings: dict[str, float] = {}

    for idx in out.sort_values("event_date").index:
        row = out.loc[idx]
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        home_elo = ratings.get(home, preseason)
        away_elo = ratings.get(away, preseason)

        injury_adj = 9.0 * float(row.get("injury_impact_diff", 0.0))
        rest_adj = 4.0 * float(row.get("rest_diff", 0.0))
        travel_adj = -0.004 * float(row.get("travel_distance_away", row.get("travel_distance", 0.0)))
        out.loc[idx, "elo_home"] = home_elo
        out.loc[idx, "elo_away"] = away_elo
        out.loc[idx, "elo_diff"] = (home_elo - away_elo) + hfa + injury_adj + rest_adj + travel_adj

        if "home_score" in out.columns and "away_score" in out.columns:
            hs = float(row.get("home_score", 0.0))
            aw = float(row.get("away_score", 0.0))
            if hs != 0.0 or aw != 0.0:
                actual = 1.0 if hs > aw else 0.0
                exp_home = 1.0 / (1.0 + 10 ** (-(home_elo + hfa - away_elo) / 400.0))
                margin = max(1.0, abs(hs - aw))
                mov_mult = math.log(margin + 1.0)
                ratings[home] = home_elo + k * mov_mult * (actual - exp_home)
                ratings[away] = away_elo + k * mov_mult * ((1.0 - actual) - (1.0 - exp_home))

    return out.fillna(0.0)

def add_market_context_features(games_df: pd.DataFrame) -> pd.DataFrame:
    out = games_df.copy()

    def _series(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    out["market_prob"] = _series("market_prob")
    out["model_prob"] = _series("model_prob")
    injury_impact_diff = _series("injury_impact_diff")
    out["model_prob"] = out["model_prob"] + (injury_impact_diff * INJURY_MODEL_PROB_WEIGHT)
    out["model_prob"] = out["model_prob"].clip(0.05, 0.95)
    out["adjusted_prob"] = _series("adjusted_prob", default=np.nan)
    needs_adjust = out["adjusted_prob"].isna()
    if needs_adjust.any():
        out.loc[needs_adjust, "adjusted_prob"] = out.loc[needs_adjust, "model_prob"]
    injury_adjustment = injury_impact_diff * INJURY_EDGE_EV_SCALING_FACTOR
    out["edge"] = (out["model_prob"] - out["market_prob"]) * EDGE_SCALING_FACTOR
    pitcher_diff = _series("pitcher_diff")
    goalie_diff = _series("goalie_diff")
    sport_series = out.get("sport", pd.Series("", index=out.index)).astype(str).str.lower()
    mlb_mask = sport_series.eq("mlb")
    pitcher_weight = pd.Series(0.03, index=out.index, dtype="float64")
    pitcher_weight.loc[mlb_mask] = 0.015
    out["adjusted_edge"] = out["edge"] + (pitcher_diff * pitcher_weight)
    out["adjusted_edge"] += goalie_diff * 0.05
    out["adjusted_edge"] = out["adjusted_edge"].clip(-0.25, 0.25)
    out["expected_value"] = _series("expected_value") + injury_adjustment
    out["open_line"] = _series("open_line") if "open_line" in out.columns else _series("spread_line")
    out["current_line"] = _series("current_line") if "current_line" in out.columns else _series("spread_line")
    out["line_movement"] = out["current_line"] - out["open_line"]
    out["public_favorite_bias_flag"] = (_series("home_odds") < -140).astype(float)
    out["public_bias_flag"] = out["public_favorite_bias_flag"]
    out["favorite_inflation_flag"] = (out["line_movement"].abs() >= 1.5).astype(float)
    out["underdog_inflation_flag"] = (out["line_movement"] <= -1.5).astype(float)
    out["bet_line"] = _series("bet_line") if "bet_line" in out.columns else out["current_line"]
    out["closing_line"] = _series("closing_line") if "closing_line" in out.columns else out["current_line"]
    out["clv_diff"] = out["closing_line"] - out["bet_line"]
    out["clv_placeholder"] = _series("clv_placeholder") if "clv_placeholder" in out.columns else out["clv_diff"]
    out["opening_line"] = out["open_line"]
    return out


def add_recent_form_features(games_df: pd.DataFrame, sport: str) -> pd.DataFrame:
    out = games_df.copy()
    out["event_date"] = pd.to_datetime(out.get("event_date"), errors="coerce", utc=True)
    out = out.sort_values("event_date").copy()

    def _rolling(team_col: str, score_col: str, opp_col: str, windows: tuple[int, ...]) -> pd.DataFrame:
        base = out[["event_date", team_col, score_col, opp_col]].rename(columns={team_col: "team", score_col: "for", opp_col: "against"}).copy()
        rev = out[["event_date", "away_team" if team_col == "home_team" else "home_team", opp_col, score_col]].rename(columns={"away_team" if team_col == "home_team" else "home_team": "team", opp_col: "for", score_col: "against"})
        games = pd.concat([base, rev], ignore_index=True).sort_values(["team", "event_date"])
        games["diff"] = pd.to_numeric(games["for"], errors="coerce").fillna(0.0) - pd.to_numeric(games["against"], errors="coerce").fillna(0.0)
        for w in windows:
            games[f"last{w}_avg"] = games.groupby("team")["diff"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        return games

    if "home_score" in out.columns and "away_score" in out.columns:
        team_form = _rolling("home_team", "home_score", "away_score", (5, 10))
        home = team_form[["event_date", "team", "last5_avg", "last10_avg"]].rename(columns={"team": "home_team", "last5_avg": "last5_home", "last10_avg": "last10_home"})
        away = team_form[["event_date", "team", "last5_avg", "last10_avg"]].rename(columns={"team": "away_team", "last5_avg": "last5_away", "last10_avg": "last10_away"})
        out = out.merge(home, on=["event_date", "home_team"], how="left").merge(away, on=["event_date", "away_team"], how="left")
        out["last5_net_rating_diff"] = pd.to_numeric(out.get("last5_away", 0.0), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("last5_home", 0.0), errors="coerce").fillna(0.0)
        out["last10_net_rating_diff"] = pd.to_numeric(out.get("last10_away", 0.0), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("last10_home", 0.0), errors="coerce").fillna(0.0)
        out["recent_goal_diff"] = out["last5_net_rating_diff"] if sport == "nhl" else 0.0
        out["recent_epa_diff"] = out["last5_net_rating_diff"] if sport == "nfl" else 0.0
    else:
        out["last5_net_rating_diff"] = 0.0
        out["last10_net_rating_diff"] = 0.0
        out["recent_goal_diff"] = 0.0
        out["recent_epa_diff"] = 0.0

    return out.fillna(0.0)


def enrich_with_context_features(games_df: pd.DataFrame, sport: str, data_root: Path) -> pd.DataFrame:
    out = add_injury_features(games_df, sport, data_root)
    out = add_travel_fatigue_features(out, sport, data_root)
    out = add_elo_features(out, sport)
    out = add_efficiency_features(out, sport, data_root)
    out = add_market_context_features(out)
    out = add_recent_form_features(out, sport)
    out["injury_impact"] = out.get("injury_impact_diff", 0.0)
    out["home_indicator"] = pd.to_numeric(out["home_indicator"], errors="coerce").fillna(1.0) if "home_indicator" in out.columns else 1.0
    return out.fillna(0.0)
