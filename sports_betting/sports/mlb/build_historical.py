"""Build MLB historical training CSV from available raw sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("sports_betting/data/historical/mlb_historical.csv")


def _american_to_prob(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else (100 / (x + 100) if x > 0 else 0.5))


def _series_or_default(df: pd.DataFrame, column: str, default: float | int) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

def build_mlb_historical() -> Path:
    """Build mlb_historical.csv from an available raw MLB game file."""
    print("[MLB] Building historical dataset...")

    possible_sources = [
        Path("sports_betting/data/raw/mlb_games.csv"),
        Path("sports_betting/data/mlb_raw.csv"),
        Path("sports_betting/data/historical/mlb_raw.csv"),
        Path("sports_betting/data/raw/mlb_historical_raw.csv"),
        Path("sports_betting/data/raw/baseball_mlb_historical.csv"),
    ]

    source_path: Path | None = None
    for p in possible_sources:
        if p.exists():
            source_path = p
            break

    if source_path is None:
        raise RuntimeError("[MLB] No raw MLB data source found. You must provide historical MLB game data.")

    df = pd.read_csv(source_path)

    required = ["home_team", "away_team", "home_score", "away_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[MLB] Missing required column: {col}")

    out = df.copy()

    out["date"] = pd.to_datetime(out.get("date", out.get("event_date")), errors="coerce", utc=True)
    out["game_id"] = out.get("game_id", out.get("event_id", out.index.astype(str))).astype(str)
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")

    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    out["elo_home"] = _series_or_default(out, "elo_home", 1500).fillna(1500)
    out["elo_away"] = _series_or_default(out, "elo_away", 1500).fillna(1500)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]

    out["rest_days_home"] = _series_or_default(out, "rest_days_home", 1).fillna(1)
    out["rest_days_away"] = _series_or_default(out, "rest_days_away", 1).fillna(1)
    out["rest_diff"] = out["rest_days_home"] - out["rest_days_away"]

    out["starter_rating_home"] = _series_or_default(out, "starter_rating_home", 0).fillna(0)
    out["starter_rating_away"] = _series_or_default(out, "starter_rating_away", 0).fillna(0)
    out["starter_rating_diff"] = out["starter_rating_home"] - out["starter_rating_away"]

    out["bullpen_rating_home"] = _series_or_default(out, "bullpen_rating_home", 0).fillna(0)
    out["bullpen_rating_away"] = _series_or_default(out, "bullpen_rating_away", 0).fillna(0)
    out["bullpen_rating_diff"] = out["bullpen_rating_home"] - out["bullpen_rating_away"]

    out["hitting_rating_home"] = _series_or_default(out, "hitting_rating_home", 0).fillna(0)
    out["hitting_rating_away"] = _series_or_default(out, "hitting_rating_away", 0).fillna(0)
    out["hitting_rating_diff"] = out["hitting_rating_home"] - out["hitting_rating_away"]

    out["injury_impact_home"] = _series_or_default(out, "injury_impact_home", 0).fillna(0)
    out["injury_impact_away"] = _series_or_default(out, "injury_impact_away", 0).fillna(0)
    out["injury_impact_diff"] = out["injury_impact_home"] - out["injury_impact_away"]

    out["home_moneyline"] = _series_or_default(out, "home_moneyline", -110).fillna(-110)
    out["spread"] = _series_or_default(out, "spread", 0).fillna(0)
    out["implied_home_prob"] = pd.to_numeric(
        out.get("implied_home_prob", _american_to_prob(out["home_moneyline"])),
        errors="coerce",
    ).fillna(0.5)
    out["spread_abs"] = out["spread"].abs()
    out["is_favorite"] = (out["home_moneyline"] < 0).astype(int)

    out["closing_moneyline_home"] = pd.to_numeric(
        out.get("closing_moneyline_home", out.get("home_odds", out["home_moneyline"])),
        errors="coerce",
    ).fillna(out["home_moneyline"])
    out["closing_moneyline_away"] = _series_or_default(out, "closing_moneyline_away", out.get("away_odds", 0) if "away_odds" in out.columns else 0).fillna(0)
    out["closing_spread_home"] = _series_or_default(out, "closing_spread_home", out.get("spread_line", out["spread"]) if "spread_line" in out.columns else out["spread"]).fillna(out["spread"])
    out["closing_total"] = _series_or_default(out, "closing_total", out.get("total_line", 0) if "total_line" in out.columns else 0).fillna(0)
    out["home_cover"] = ((out["home_score"] - out["away_score"] + out["closing_spread_home"]) > 0).astype(int)
    out["over_hit"] = ((out["home_score"] + out["away_score"]) > out["closing_total"]).astype(int)

    out["travel_distance_home"] = _series_or_default(out, "travel_distance_home", 0).fillna(0)
    out["travel_distance_away"] = _series_or_default(out, "travel_distance_away", 0).fillna(0)
    out["travel_distance"] = out["travel_distance_home"] + out["travel_distance_away"]
    out["travel_fatigue_diff"] = out["travel_distance_home"] - out["travel_distance_away"]

    out["home_split_home"] = _series_or_default(out, "home_split_home", 0).fillna(0)
    out["home_split_away"] = _series_or_default(out, "home_split_away", 0).fillna(0)
    out["home_split_diff"] = out["home_split_home"] - out["home_split_away"]

    out["recent_form_home"] = _series_or_default(out, "recent_form_home", 0).fillna(0)
    out["recent_form_away"] = _series_or_default(out, "recent_form_away", 0).fillna(0)
    out["recent_form_diff"] = out["recent_form_home"] - out["recent_form_away"]

    required_cols = [
        "date",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "home_cover",
        "over_hit",
        "closing_moneyline_home",
        "closing_moneyline_away",
        "closing_spread_home",
        "closing_total",
        "elo_home",
        "elo_away",
        "elo_diff",
        "rest_days_home",
        "rest_days_away",
        "rest_diff",
        "travel_distance_home",
        "travel_distance_away",
        "travel_distance",
        "travel_fatigue_diff",
        "injury_impact_home",
        "injury_impact_away",
        "injury_impact_diff",
        "starter_rating_home",
        "starter_rating_away",
        "starter_rating_diff",
        "bullpen_rating_home",
        "bullpen_rating_away",
        "bullpen_rating_diff",
        "hitting_rating_home",
        "hitting_rating_away",
        "hitting_rating_diff",
        "home_split_home",
        "home_split_away",
        "home_split_diff",
        "recent_form_home",
        "recent_form_away",
        "recent_form_diff",
        "home_moneyline",
        "spread",
        "implied_home_prob",
        "spread_abs",
        "is_favorite",
    ]

    out = out[required_cols].dropna(subset=["home_team", "away_team", "home_score", "away_score", "home_win"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[MLB] Historical dataset created: {len(out)} rows")
    return OUTPUT_PATH


def build_mlb_historical_csv() -> Path:
    """Backward-compatible alias for existing callers."""
    return build_mlb_historical()


if __name__ == "__main__":
    path = build_mlb_historical()
    print(f"[MLB] historical dataset saved: {path}")
