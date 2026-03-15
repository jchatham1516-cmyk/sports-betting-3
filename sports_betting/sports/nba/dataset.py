"""NBA historical dataset builder used by model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sports_betting.sports.nba.features import build_nba_features

DATA_ROOT = Path("sports_betting/data")
HISTORICAL_OUTPUT_PATH = DATA_ROOT / "historical" / "nba_historical.csv"

NBA_HISTORICAL_COLUMNS = [
    "date", "season", "game_id", "home_team", "away_team",
    "home_score", "away_score", "home_win", "home_cover", "over_hit", "margin", "total_points",
    "closing_moneyline_home", "closing_moneyline_away", "closing_spread_home", "closing_total", "market_prob_home", "market_prob_away",
    "elo_home", "elo_away", "elo_diff",
    "rest_days_home", "rest_days_away", "rest_diff", "back_to_back_home", "back_to_back_away", "three_in_four_home", "three_in_four_away",
    "travel_distance_home", "travel_distance_away", "timezone_shift_home", "timezone_shift_away", "road_trip_length_home", "road_trip_length_away", "travel_fatigue_diff",
    "injury_impact_home", "injury_impact_away", "injury_impact_diff", "starter_out_count_home", "starter_out_count_away", "star_player_out_home", "star_player_out_away",
    "off_rating_home", "off_rating_away", "def_rating_home", "def_rating_away", "net_rating_home", "net_rating_away", "net_rating_diff",
    "pace_home", "pace_away", "pace_diff", "true_shooting_home", "true_shooting_away", "true_shooting_diff",
    "turnover_rate_home", "turnover_rate_away", "turnover_rate_diff", "rebound_rate_home", "rebound_rate_away", "rebound_rate_diff",
    "last5_net_rating_home", "last5_net_rating_away", "last5_net_rating_diff", "last10_net_rating_home", "last10_net_rating_away", "last10_net_rating_diff",
]

_IDENTITY_COLUMNS = {"date", "season", "game_id", "home_team", "away_team"}


def _coalesce_numeric(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _american_to_prob(odds: pd.Series) -> pd.Series:
    numeric_odds = pd.to_numeric(odds, errors="coerce")
    probs = np.where(
        numeric_odds < 0,
        (-numeric_odds) / ((-numeric_odds) + 100.0),
        np.where(numeric_odds > 0, 100.0 / (numeric_odds + 100.0), np.nan),
    )
    return pd.Series(probs, index=odds.index, dtype="float64")


def _derive_team_rolling_metrics(df: pd.DataFrame, team_col: str, opponent_col: str, score_for: str, score_against: str) -> pd.DataFrame:
    """Compute leakage-safe rolling net rating proxies from prior games only."""
    work = df[["date", team_col, opponent_col, score_for, score_against]].copy()
    work = work.sort_values("date")

    work["game_net"] = pd.to_numeric(work[score_for], errors="coerce").fillna(0.0) - pd.to_numeric(work[score_against], errors="coerce").fillna(0.0)

    grouped = work.groupby(team_col, group_keys=False)
    work["last5_net"] = grouped["game_net"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    work["last10_net"] = grouped["game_net"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    work["rest_days"] = grouped["date"].transform(lambda s: s.diff().dt.days.clip(lower=0, upper=10))
    work["back_to_back"] = (work["rest_days"].fillna(3) <= 1).astype(int)
    work["three_in_four"] = grouped["date"].transform(
        lambda s: s.diff().dt.days.fillna(10).rolling(3, min_periods=3).sum().le(4).astype(int)
    )

    return work[["date", team_col, opponent_col, "last5_net", "last10_net", "rest_days", "back_to_back", "three_in_four"]]




def _synthetic_placeholder(rows: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-10-01", periods=rows, freq="D")
    teams = [f"TEAM_{i:02d}" for i in range(1, 31)]
    home_score = rng.integers(90, 130, size=rows)
    away_score = rng.integers(88, 126, size=rows)
    ml_home = rng.integers(-220, 190, size=rows)
    spread = rng.normal(-2.0, 6.0, size=rows)
    total = rng.normal(227.0, 12.0, size=rows)
    return pd.DataFrame({
        "date": dates,
        "game_id": [f"NBA-SYN-{idx:07d}" for idx in range(rows)],
        "home_team": rng.choice(teams, size=rows),
        "away_team": rng.choice(teams, size=rows),
        "home_score": home_score,
        "away_score": away_score,
        "closing_moneyline_home": ml_home,
        "closing_moneyline_away": -ml_home,
        "closing_spread_home": spread.round(1),
        "closing_total": total.round(1),
    })

def _load_source_dataframe(input_path: Path | None) -> pd.DataFrame:
    candidates = [
        input_path,
        DATA_ROOT / "raw" / "nba_historical_raw.csv",
        DATA_ROOT / "historical" / "nba_historical.csv",
    ]
    source = next((path for path in candidates if path and path.exists()), None)
    if source is None:
        return _synthetic_placeholder()
    return pd.read_csv(source)


def build_nba_historical_dataset(input_path: Path | None = None, output_path: Path = HISTORICAL_OUTPUT_PATH) -> Path:
    """Build and persist NBA historical dataset with pre-game feature semantics."""
    df = _load_source_dataframe(input_path)

    df["date"] = pd.to_datetime(df.get("date", df.get("event_date")), errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["season"] = _coalesce_numeric(df, ["season"], default=np.nan)
    inferred_season = np.where(df["date"].dt.month >= 10, df["date"].dt.year, df["date"].dt.year - 1)
    df["season"] = df["season"].fillna(pd.Series(inferred_season, index=df.index)).astype(int)

    if "game_id" not in df.columns:
        df["game_id"] = [f"NBA-{idx:07d}" for idx in range(len(df))]
    df["game_id"] = df["game_id"].astype(str).replace("", np.nan)
    df["game_id"] = df["game_id"].fillna(pd.Series([f"NBA-{idx:07d}" for idx in range(len(df))], index=df.index))

    df["home_team"] = df.get("home_team", "HOME").astype(str)
    df["away_team"] = df.get("away_team", "AWAY").astype(str)

    df["home_score"] = _coalesce_numeric(df, ["home_score"], default=0.0).fillna(0.0)
    df["away_score"] = _coalesce_numeric(df, ["away_score"], default=0.0).fillna(0.0)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]

    df["closing_moneyline_home"] = _coalesce_numeric(df, ["closing_moneyline_home", "home_odds"], default=-110.0).fillna(-110.0)
    df["closing_moneyline_away"] = _coalesce_numeric(df, ["closing_moneyline_away", "away_odds"], default=-110.0).fillna(-110.0)
    df["closing_spread_home"] = _coalesce_numeric(df, ["closing_spread_home", "spread_line"], default=0.0).fillna(0.0)
    df["closing_total"] = _coalesce_numeric(df, ["closing_total", "total_line"], default=220.0).fillna(220.0)

    df["home_win"] = (df["margin"] > 0).astype(int)
    df["home_cover"] = ((df["margin"] + df["closing_spread_home"]) > 0).astype(int)
    df["over_hit"] = (df["total_points"] > df["closing_total"]).astype(int)

    df["market_prob_home"] = _american_to_prob(df["closing_moneyline_home"]).fillna(0.5)
    df["market_prob_away"] = _american_to_prob(df["closing_moneyline_away"]).fillna(0.5)

    # Rolling form and schedule features from prior games only.
    home_hist = _derive_team_rolling_metrics(df, "home_team", "away_team", "home_score", "away_score")
    away_hist = _derive_team_rolling_metrics(df, "away_team", "home_team", "away_score", "home_score")

    df["last5_net_rating_home"] = home_hist["last5_net"].fillna(0.0).values
    df["last10_net_rating_home"] = home_hist["last10_net"].fillna(0.0).values
    df["rest_days_home"] = home_hist["rest_days"].fillna(3.0).values
    df["back_to_back_home"] = home_hist["back_to_back"].fillna(0.0).values
    df["three_in_four_home"] = home_hist["three_in_four"].fillna(0.0).values

    df["last5_net_rating_away"] = away_hist["last5_net"].fillna(0.0).values
    df["last10_net_rating_away"] = away_hist["last10_net"].fillna(0.0).values
    df["rest_days_away"] = away_hist["rest_days"].fillna(3.0).values
    df["back_to_back_away"] = away_hist["back_to_back"].fillna(0.0).values
    df["three_in_four_away"] = away_hist["three_in_four"].fillna(0.0).values

    # Fallbacks for optional upstream enriched stats.
    for column, fallback in {
        "elo_home": 1500.0,
        "elo_away": 1500.0,
        "travel_distance_home": 0.0,
        "travel_distance_away": 0.0,
        "timezone_shift_home": 0.0,
        "timezone_shift_away": 0.0,
        "road_trip_length_home": 0.0,
        "road_trip_length_away": 0.0,
        "injury_impact_home": 0.0,
        "injury_impact_away": 0.0,
        "starter_out_count_home": 0.0,
        "starter_out_count_away": 0.0,
        "star_player_out_home": 0.0,
        "star_player_out_away": 0.0,
        "off_rating_home": 0.0,
        "off_rating_away": 0.0,
        "def_rating_home": 0.0,
        "def_rating_away": 0.0,
        "net_rating_home": 0.0,
        "net_rating_away": 0.0,
        "pace_home": 0.0,
        "pace_away": 0.0,
        "true_shooting_home": 0.0,
        "true_shooting_away": 0.0,
        "turnover_rate_home": 0.0,
        "turnover_rate_away": 0.0,
        "rebound_rate_home": 0.0,
        "rebound_rate_away": 0.0,
    }.items():
        df[column] = _coalesce_numeric(df, [column], default=fallback).fillna(fallback)

    df = build_nba_features(df)
    df["elo_home"] = _coalesce_numeric(df, ["elo_home"], 1500.0).fillna(1500.0)
    df["elo_away"] = _coalesce_numeric(df, ["elo_away"], 1500.0).fillna(1500.0)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["injury_impact_diff"] = df["injury_impact_home"] - df["injury_impact_away"]
    df["net_rating_diff"] = df["net_rating_home"] - df["net_rating_away"]
    df["pace_diff"] = df["pace_home"] - df["pace_away"]
    df["true_shooting_diff"] = df["true_shooting_home"] - df["true_shooting_away"]
    df["turnover_rate_diff"] = df["turnover_rate_home"] - df["turnover_rate_away"]
    df["rebound_rate_diff"] = df["rebound_rate_home"] - df["rebound_rate_away"]
    df["last5_net_rating_diff"] = df["last5_net_rating_home"] - df["last5_net_rating_away"]
    df["last10_net_rating_diff"] = df["last10_net_rating_home"] - df["last10_net_rating_away"]

    df["travel_fatigue_diff"] = (
        (df["travel_distance_away"] + 250.0 * df["timezone_shift_away"] + 150.0 * df["road_trip_length_away"])
        - (df["travel_distance_home"] + 250.0 * df["timezone_shift_home"] + 150.0 * df["road_trip_length_home"])
    )

    for column in NBA_HISTORICAL_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        if column not in _IDENTITY_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    final = df.reindex(columns=NBA_HISTORICAL_COLUMNS)
    final["date"] = pd.to_datetime(final["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    return output_path
