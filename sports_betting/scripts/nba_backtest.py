"""NBA historical backtest report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from sports_betting.sports.common.odds import american_to_implied_probability, expected_value
from sports_betting.sports.nba.simple_model import predict, train_runtime_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "sports_betting" / "data" / "historical" / "nba_historical.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "outputs"


def _profit(prob_result: int, odds: float) -> float:
    if prob_result != 1:
        return -1.0
    if odds > 0:
        return float(odds) / 100.0
    if odds < 0:
        return 100.0 / abs(float(odds))
    return 0.0


def _bucket(value: float, buckets: list[tuple[float, float, str]]) -> str:
    for lo, hi, label in buckets:
        if value >= lo and value < hi:
            return label
    return buckets[-1][2]


def run_nba_backtest(
    historical_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    min_train_rows: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Train once on the first 80% of nba_historical.csv and report holdout performance.

    The historical file currently has moneyline outcomes, so spread/total rows are
    emitted only when matching result columns exist in future datasets.
    """

    historical_path = Path(historical_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not historical_path.exists():
        raise FileNotFoundError(f"NBA historical file not found: {historical_path}")

    df = pd.read_csv(historical_path)
    if df.empty:
        raise ValueError("NBA historical file is empty")
    if "home_win" not in df.columns:
        raise ValueError("NBA backtest requires home_win in historical data")
    if "home_moneyline" not in df.columns and "closing_moneyline_home" in df.columns:
        df["home_moneyline"] = df["closing_moneyline_home"]
    if "home_moneyline" not in df.columns:
        raise ValueError("NBA backtest requires home_moneyline or closing_moneyline_home")

    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")
    before_clean = len(df)
    df = df.dropna(subset=["home_moneyline", "home_win"]).copy()
    df = df[df["home_moneyline"].ne(0)].copy()
    skipped_rows = before_clean - len(df)
    if skipped_rows:
        print(f"[NBA BACKTEST] skipped {skipped_rows} rows with missing/invalid moneyline or result")
    if df.empty:
        raise ValueError("NBA backtest has no valid rows after dropping missing moneyline/result values")
    df["home_win"] = df["home_win"].astype(int)

    if "date" in df.columns:
        df = df.assign(_date=pd.to_datetime(df["date"], errors="coerce")).sort_values(["_date"]).drop(columns=["_date"])
    split_idx = max(min_train_rows, int(len(df) * 0.8))
    if split_idx >= len(df):
        split_idx = max(1, len(df) // 2)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty:
        train_df = df.copy()
        test_df = df.copy()

    model = train_runtime_model(train_df)
    if model is None:
        raise ValueError("NBA backtest could not train a runtime model")

    probs = predict(model, test_df)
    test_df["model_probability"] = probs
    test_df["market_probability"] = pd.to_numeric(test_df["home_moneyline"], errors="coerce").apply(american_to_implied_probability).fillna(0.5)
    test_df["edge"] = test_df["model_probability"] - test_df["market_probability"]
    valid_odds = pd.to_numeric(test_df["home_moneyline"], errors="coerce").fillna(0.0)
    test_df["expected_value"] = [expected_value(float(p), int(o)) if float(o) != 0.0 else 0.0 for p, o in zip(test_df["model_probability"], valid_odds)]
    test_df["home_win"] = pd.to_numeric(test_df["home_win"], errors="coerce").fillna(0).astype(int)
    test_df["bet_result_units"] = [_profit(int(w), float(o)) if float(o) != 0.0 else 0.0 for w, o in zip(test_df["home_win"], valid_odds)]
    test_df["probability_bucket"] = test_df["model_probability"].apply(
        lambda v: _bucket(float(v), [(0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), (0.65, 1.01, "65%+")])
    )
    test_df["edge_bucket"] = test_df["edge"].apply(
        lambda v: _bucket(float(max(v, 0.0)), [(0.00, 0.02, "0-2%"), (0.02, 0.04, "2-4%"), (0.04, 0.06, "4-6%"), (0.06, 1.01, "6%+")])
    )
    test_df["market"] = "moneyline"

    total_games = int(len(test_df))
    wins = int(test_df["home_win"].sum())
    profit_loss = float(test_df["bet_result_units"].sum())
    summary = {
        "total_games_tested": total_games,
        "win_rate": float(wins / total_games) if total_games else 0.0,
        "roi": float(profit_loss / total_games) if total_games else 0.0,
        "profit_loss": profit_loss,
        "validation_log_loss": float(log_loss(test_df["home_win"], np.clip(probs, 0.001, 0.999), labels=[0, 1])) if test_df["home_win"].nunique() > 1 else None,
        "validation_brier_score": float(brier_score_loss(test_df["home_win"], np.clip(probs, 0.001, 0.999))),
        "train_rows": int(len(train_df)),
        "test_rows": total_games,
        "split_method": "chronological_80_20" if "date" in df.columns else "row_order_80_20",
    }

    sections: list[pd.DataFrame] = []
    for name, group_col in [
        ("calibration_bucket", "probability_bucket"),
        ("market", "market"),
        ("edge_bucket", "edge_bucket"),
    ]:
        section = test_df.groupby(group_col, observed=False).agg(
            total_games=("home_win", "size"),
            win_rate=("home_win", "mean"),
            avg_model_probability=("model_probability", "mean"),
            avg_edge=("edge", "mean"),
            roi=("bet_result_units", "mean"),
            profit_loss=("bet_result_units", "sum"),
        ).reset_index().rename(columns={group_col: "bucket"})
        section.insert(0, "section", name)
        sections.append(section)

    report_df = pd.concat(sections, ignore_index=True)
    report_df.to_csv(output_dir / "nba_backtest_report.csv", index=False)
    (output_dir / "nba_backtest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[NBA BACKTEST] wrote {output_dir / 'nba_backtest_report.csv'}")
    print(f"[NBA BACKTEST] wrote {output_dir / 'nba_backtest_summary.json'}")
    print(f"[NBA BACKTEST] summary: {summary}")
    return report_df, summary


if __name__ == "__main__":
    run_nba_backtest()
