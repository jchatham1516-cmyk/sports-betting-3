"""Build MLB historical training CSV from available raw sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports_betting.scripts.data_io import ROOT
from sports_betting.sports.mlb.features import build_mlb_features
from sports_betting.sports.mlb.schema import MLB_HISTORICAL_COLUMNS


RAW_CANDIDATE_PATHS = [
    ROOT / "raw" / "mlb_historical_raw.csv",
    ROOT / "raw" / "mlb_games.csv",
    ROOT / "raw" / "baseball_mlb_historical.csv",
]
OUTPUT_PATH = ROOT / "historical" / "mlb_historical.csv"


def _american_to_prob(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else (100 / (x + 100) if x > 0 else 0.5))


def _find_raw_source() -> Path:
    for path in RAW_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise RuntimeError("[MLB] No raw historical source found to build mlb_historical.csv")


def build_mlb_historical_csv() -> Path:
    source_path = _find_raw_source()
    raw = pd.read_csv(source_path)
    out = raw.copy()

    out["date"] = pd.to_datetime(out.get("date", out.get("event_date")), errors="coerce", utc=True)
    out["game_id"] = out.get("game_id", out.get("event_id", out.index.astype(str))).astype(str)
    out["home_team"] = out.get("home_team", "").astype(str)
    out["away_team"] = out.get("away_team", "").astype(str)
    out["home_score"] = pd.to_numeric(out.get("home_score", 0), errors="coerce").fillna(0)
    out["away_score"] = pd.to_numeric(out.get("away_score", 0), errors="coerce").fillna(0)
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    out["closing_moneyline_home"] = pd.to_numeric(
        out.get("closing_moneyline_home", out.get("home_odds", out.get("home_moneyline", 0))),
        errors="coerce",
    ).fillna(0)
    out["closing_moneyline_away"] = pd.to_numeric(
        out.get("closing_moneyline_away", out.get("away_odds", 0)),
        errors="coerce",
    ).fillna(0)
    out["closing_spread_home"] = pd.to_numeric(
        out.get("closing_spread_home", out.get("spread_line", out.get("spread", 0))),
        errors="coerce",
    ).fillna(0)
    out["closing_total"] = pd.to_numeric(
        out.get("closing_total", out.get("total_line", 0)),
        errors="coerce",
    ).fillna(0)
    out["home_cover"] = ((out["home_score"] - out["away_score"] + out["closing_spread_home"]) > 0).astype(int)
    out["over_hit"] = ((out["home_score"] + out["away_score"]) > out["closing_total"]).astype(int)
    out["home_moneyline"] = out["closing_moneyline_home"]
    out["spread"] = out["closing_spread_home"]
    out["implied_home_prob"] = _american_to_prob(out["home_moneyline"])
    out["spread_abs"] = out["spread"].abs()
    out["is_favorite"] = (out["home_moneyline"] < 0).astype(int)

    out = build_mlb_features(out)

    for col in MLB_HISTORICAL_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out = out.reindex(columns=MLB_HISTORICAL_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_mlb_historical_csv()
    print(f"[MLB] historical dataset saved: {path}")
