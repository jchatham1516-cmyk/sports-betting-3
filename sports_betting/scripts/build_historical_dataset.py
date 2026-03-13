"""Build enriched historical datasets for NBA/NFL/NHL with robust fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sports_betting.scripts.data_io import ROOT, historical_file_path
from sports_betting.sports.common.feature_engineering import enrich_with_context_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enriched historical datasets.")
    parser.add_argument("--sports", nargs="+", choices=["nba", "nfl", "nhl"], default=["nba", "nfl", "nhl"])
    parser.add_argument("--input-dir", default=str(ROOT / "raw"), help="Directory containing <sport>_historical_raw.csv")
    return parser.parse_args()


def _default_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hs = pd.to_numeric(out.get("home_score", 0), errors="coerce").fillna(0)
    aw = pd.to_numeric(out.get("away_score", 0), errors="coerce").fillna(0)
    spread = pd.to_numeric(out.get("closing_spread_home", out.get("spread_line", 0)), errors="coerce").fillna(0)
    total = pd.to_numeric(out.get("closing_total", out.get("total_line", 0)), errors="coerce").fillna(0)
    out["home_win"] = ((hs - aw) > 0).astype(int)
    out["home_cover"] = ((hs - aw + spread) > 0).astype(int)
    out["over_hit"] = ((hs + aw) > total).astype(int)
    return out


def build_historical_dataset(sport: str, input_dir: Path) -> Path:
    source = input_dir / f"{sport}_historical_raw.csv"
    if not source.exists():
        source = historical_file_path(sport)
    if not source.exists():
        raise RuntimeError(f"[{sport.upper()}] No historical source found at {source}")

    df = pd.read_csv(source)
    if len(df) < 500:
        raise RuntimeError(f"[{sport.upper()}] historical source is too small ({len(df)} rows). Provide a larger multi-season dataset.")
    required_identity = {"home_team", "away_team"}
    missing_identity = [col for col in required_identity if col not in df.columns]
    if missing_identity:
        raise RuntimeError(f"[{sport.upper()}] historical source missing required columns: {', '.join(missing_identity)}")
    df["event_date"] = pd.to_datetime(df.get("date", df.get("event_date")), errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)

    # Map line aliases and ensure required schema.
    df["closing_moneyline_home"] = pd.to_numeric(df.get("closing_moneyline_home", df.get("home_odds", 0)), errors="coerce").fillna(0)
    df["closing_moneyline_away"] = pd.to_numeric(df.get("closing_moneyline_away", df.get("away_odds", 0)), errors="coerce").fillna(0)
    df["closing_spread_home"] = pd.to_numeric(df.get("closing_spread_home", df.get("spread_line", 0)), errors="coerce").fillna(0)
    df["closing_total"] = pd.to_numeric(df.get("closing_total", df.get("total_line", 0)), errors="coerce").fillna(0)
    df["spread_line"] = pd.to_numeric(df.get("spread_line", df["closing_spread_home"]), errors="coerce").fillna(0)
    df["total_line"] = pd.to_numeric(df.get("total_line", df["closing_total"]), errors="coerce").fillna(0)

    if any(c not in df.columns for c in ["home_win", "home_cover", "over_hit"]):
        df = _default_labels(df)

    enriched = enrich_with_context_features(df, sport, ROOT)
    out_path = historical_file_path(sport)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    for sport in args.sports:
        path = build_historical_dataset(sport, input_dir)
        print(f"[{sport.upper()}] wrote enriched historical dataset -> {path}")


if __name__ == "__main__":
    main()
