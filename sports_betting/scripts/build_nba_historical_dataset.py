"""Build a structured NBA historical dataset with robust fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sports_betting/data")

NBA_HISTORICAL_COLUMNS = [
    "date", "season", "game_id", "home_team", "away_team",
    "home_score", "away_score", "home_win", "home_cover", "over_hit", "margin", "total_points",
    "closing_moneyline_home", "closing_moneyline_away", "closing_spread_home", "closing_total", "market_prob_home", "market_prob_away",
    "elo_home", "elo_away", "elo_diff",
    "rest_days_home", "rest_days_away", "rest_diff", "back_to_back_home", "back_to_back_away", "three_in_four_home", "three_in_four_away",
    "travel_distance_home", "travel_distance_away", "timezone_shift_home", "timezone_shift_away", "road_trip_length_home", "road_trip_length_away", "travel_fatigue_diff",
    "injury_impact_home", "injury_impact_away", "injury_impact_diff", "starter_out_count_home", "starter_out_count_away", "star_player_out_home", "star_player_out_away",
    "offensive_injury_weight_home", "offensive_injury_weight_away", "defensive_injury_weight_home", "defensive_injury_weight_away",
    "off_rating_home", "off_rating_away", "def_rating_home", "def_rating_away", "net_rating_home", "net_rating_away", "net_rating_diff",
    "pace_home", "pace_away", "pace_diff", "true_shooting_home", "true_shooting_away", "true_shooting_diff", "turnover_rate_home", "turnover_rate_away", "turnover_rate_diff",
    "rebound_rate_home", "rebound_rate_away", "rebound_rate_diff", "top_rotation_efficiency_home", "top_rotation_efficiency_away", "top_rotation_eff_diff",
    "last5_net_rating_home", "last5_net_rating_away", "last5_net_rating_diff", "last10_net_rating_home", "last10_net_rating_away", "last10_net_rating_diff",
    "recent_off_rating_diff", "recent_def_rating_diff", "recent_total_trend",
    "opening_moneyline_home", "opening_spread_home", "opening_total", "line_movement_spread", "line_movement_total", "clv_placeholder",
]

NUMERIC_DEFAULT_ZERO = [c for c in NBA_HISTORICAL_COLUMNS if c not in {"date", "season", "game_id", "home_team", "away_team"}]


def _to_num(df: pd.DataFrame, target: str, candidates: list[str], default: float = 0.0) -> None:
    for c in candidates:
        if c in df.columns:
            df[target] = pd.to_numeric(df[c], errors="coerce").fillna(default)
            return
    df[target] = default


def _american_to_prob(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce").fillna(0.0)
    fav = np.where(o < 0, (-o) / ((-o) + 100.0), 0.0)
    dog = np.where(o > 0, 100.0 / (o + 100.0), 0.5)
    return pd.Series(np.where(o < 0, fav, dog), index=odds.index)


def _placeholder_dataset(rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-10-01", periods=rows, freq="D")
    home = rng.integers(90, 130, size=rows)
    away = rng.integers(88, 126, size=rows)
    spread = rng.normal(-2.0, 6.0, size=rows).round(1)
    total = rng.normal(226.0, 12.0, size=rows).round(1)
    ml_home = rng.integers(-220, 180, size=rows)
    ml_away = -ml_home
    teams = [f"TEAM_{i:02d}" for i in range(1, 31)]
    return pd.DataFrame(
        {
            "date": dates,
            "season": np.where(dates.month >= 10, dates.year, dates.year - 1),
            "game_id": [f"NBA-{i:06d}" for i in range(rows)],
            "home_team": rng.choice(teams, size=rows),
            "away_team": rng.choice(teams, size=rows),
            "home_score": home,
            "away_score": away,
            "closing_moneyline_home": ml_home,
            "closing_moneyline_away": ml_away,
            "closing_spread_home": spread,
            "closing_total": total,
            "opening_moneyline_home": ml_home + rng.integers(-10, 11, size=rows),
            "opening_spread_home": spread + rng.normal(0.0, 0.9, size=rows),
            "opening_total": total + rng.normal(0.0, 1.3, size=rows),
        }
    )


def build_nba_historical_dataset(input_dir: Path | None = None) -> Path:
    input_dir = input_dir or (ROOT / "raw")
    candidates = [
        input_dir / "nba_historical_raw.csv",
        ROOT / "raw" / "nba_historical_raw.csv",
        ROOT / "historical" / "nba_historical.csv",
    ]
    source = next((p for p in candidates if p.exists()), None)
    df = pd.read_csv(source) if source else _placeholder_dataset()

    if "date" not in df.columns and "event_date" in df.columns:
        df["date"] = df["event_date"]
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    if df["date"].isna().all():
        df["date"] = pd.date_range("2019-10-01", periods=len(df), freq="D")
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    season_fallback = pd.Series(np.where(df["date"].dt.month >= 10, df["date"].dt.year, df["date"].dt.year - 1), index=df.index)
    df["season"] = df["season"].where(df["season"].notna(), season_fallback).astype(int)

    df["game_id"] = df.get("game_id", "")
    blank_game_id = df["game_id"].astype(str).str.len() == 0
    df.loc[blank_game_id, "game_id"] = [f"NBA-{i:06d}" for i in range(blank_game_id.sum())]

    df["home_team"] = df.get("home_team", "HOME")
    df["away_team"] = df.get("away_team", "AWAY")
    _to_num(df, "home_score", ["home_score"])
    _to_num(df, "away_score", ["away_score"])

    _to_num(df, "closing_moneyline_home", ["closing_moneyline_home", "home_odds"], default=-110)
    _to_num(df, "closing_moneyline_away", ["closing_moneyline_away", "away_odds"], default=-110)
    _to_num(df, "closing_spread_home", ["closing_spread_home", "spread_line"], default=0)
    _to_num(df, "closing_total", ["closing_total", "total_line"], default=220)

    _to_num(df, "opening_moneyline_home", ["opening_moneyline_home"], default=df["closing_moneyline_home"].mean())
    _to_num(df, "opening_spread_home", ["opening_spread_home"], default=0)
    _to_num(df, "opening_total", ["opening_total"], default=df["closing_total"].mean())

    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["home_cover"] = ((df["margin"] + df["closing_spread_home"]) > 0).astype(int)
    df["over_hit"] = (df["total_points"] > df["closing_total"]).astype(int)

    df["market_prob_home"] = _american_to_prob(df["closing_moneyline_home"])
    df["market_prob_away"] = _american_to_prob(df["closing_moneyline_away"])

    _to_num(df, "elo_home", ["elo_home"], default=1500)
    _to_num(df, "elo_away", ["elo_away"], default=1500)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    for col in [
        "rest_days_home", "rest_days_away", "back_to_back_home", "back_to_back_away", "three_in_four_home", "three_in_four_away",
        "travel_distance_home", "travel_distance_away", "timezone_shift_home", "timezone_shift_away", "road_trip_length_home", "road_trip_length_away",
        "injury_impact_home", "injury_impact_away", "starter_out_count_home", "starter_out_count_away", "star_player_out_home", "star_player_out_away",
        "offensive_injury_weight_home", "offensive_injury_weight_away", "defensive_injury_weight_home", "defensive_injury_weight_away",
        "off_rating_home", "off_rating_away", "def_rating_home", "def_rating_away", "net_rating_home", "net_rating_away", "pace_home", "pace_away",
        "true_shooting_home", "true_shooting_away", "turnover_rate_home", "turnover_rate_away", "rebound_rate_home", "rebound_rate_away",
        "top_rotation_efficiency_home", "top_rotation_efficiency_away", "last5_net_rating_home", "last5_net_rating_away", "last10_net_rating_home", "last10_net_rating_away",
        "line_movement_spread", "line_movement_total", "clv_placeholder",
    ]:
        _to_num(df, col, [col])

    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["travel_fatigue_diff"] = (
        (df["travel_distance_away"] + 250.0 * df["timezone_shift_away"] + 150.0 * df["road_trip_length_away"])
        - (df["travel_distance_home"] + 250.0 * df["timezone_shift_home"] + 150.0 * df["road_trip_length_home"])
    )
    df["injury_impact_diff"] = df["injury_impact_home"] - df["injury_impact_away"]
    df["net_rating_diff"] = df["net_rating_home"] - df["net_rating_away"]
    df["pace_diff"] = df["pace_home"] - df["pace_away"]
    df["true_shooting_diff"] = df["true_shooting_home"] - df["true_shooting_away"]
    df["turnover_rate_diff"] = df["turnover_rate_home"] - df["turnover_rate_away"]
    df["rebound_rate_diff"] = df["rebound_rate_home"] - df["rebound_rate_away"]
    df["top_rotation_eff_diff"] = df["top_rotation_efficiency_home"] - df["top_rotation_efficiency_away"]
    df["last5_net_rating_diff"] = df["last5_net_rating_home"] - df["last5_net_rating_away"]
    df["last10_net_rating_diff"] = df["last10_net_rating_home"] - df["last10_net_rating_away"]

    _to_num(df, "recent_off_rating_diff", ["recent_off_rating_diff", "off_rating_diff"])
    _to_num(df, "recent_def_rating_diff", ["recent_def_rating_diff", "def_rating_diff"])
    _to_num(df, "recent_total_trend", ["recent_total_trend", "total_points"])

    df["line_movement_spread"] = df["closing_spread_home"] - df["opening_spread_home"]
    df["line_movement_total"] = df["closing_total"] - df["opening_total"]

    for c in NUMERIC_DEFAULT_ZERO:
        if c not in df.columns:
            df[c] = 0.0
        if c not in {"home_win", "home_cover", "over_hit", "season"}:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    df["home_cover"] = pd.to_numeric(df["home_cover"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    df["over_hit"] = pd.to_numeric(df["over_hit"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)

    final_df = df.reindex(columns=NBA_HISTORICAL_COLUMNS, fill_value=0.0)
    final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_path = ROOT / "historical" / "nba_historical.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured NBA historical dataset.")
    parser.add_argument("--input-dir", default=str(ROOT / "raw"), help="Directory containing nba_historical_raw.csv")
    args = parser.parse_args()
    out_path = build_nba_historical_dataset(Path(args.input_dir))
    print(f"[NBA] wrote historical dataset -> {out_path}")


if __name__ == "__main__":
    main()
