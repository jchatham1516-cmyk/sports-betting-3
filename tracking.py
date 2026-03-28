import os
from datetime import datetime

import pandas as pd

TRACKING_FILE = "data/tracking/bet_history.csv"
REQUIRED_TRACKING_COLUMNS = [
    "result",
    "profit",
    "closing_odds",
    "closing_line",
    "bet_timestamp",
]


def _american_to_payout(odds: float) -> float:
    if pd.isna(odds):
        return 0.0
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return 0.0


def _ensure_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in REQUIRED_TRACKING_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    if "result" in out.columns:
        out["result"] = out["result"].fillna("pending")
    if "profit" in out.columns:
        out["profit"] = pd.to_numeric(out["profit"], errors="coerce").fillna(0.0)
    return out


def _recompute_profit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "units" not in out.columns:
        out["units"] = 1.0
    out["units"] = pd.to_numeric(out["units"], errors="coerce").fillna(0.0)

    payout_series = pd.Series(0.0, index=out.index, dtype="float64")
    if "payout" in out.columns:
        payout_series = pd.to_numeric(out["payout"], errors="coerce").fillna(0.0)
    elif "odds" in out.columns:
        payout_series = out["odds"].apply(_american_to_payout)

    win_mask = out["result"].eq("win")
    loss_mask = out["result"].eq("loss")

    out.loc[win_mask, "profit"] = out.loc[win_mask, "units"] * payout_series.loc[win_mask]
    out.loc[loss_mask, "profit"] = -out.loc[loss_mask, "units"]
    out.loc[~(win_mask | loss_mask), "profit"] = 0.0
    return out


def print_performance_summary(df: pd.DataFrame, exported_bets_count: int | None = None) -> None:
    settled = df[df["result"].isin(["win", "loss"])].copy()
    total_bets = int(exported_bets_count) if exported_bets_count is not None else len(settled)
    wins = int(settled["result"].eq("win").sum())
    settled_bets_count = len(settled)
    win_rate = (wins / settled_bets_count) if settled_bets_count else 0.0
    total_profit = float(settled["profit"].sum()) if settled_bets_count else 0.0
    total_units_wagered = float(settled["units"].sum()) if settled_bets_count and "units" in settled.columns else 0.0
    roi = (total_profit / total_units_wagered) if total_units_wagered else 0.0

    print("\n[PERFORMANCE SUMMARY]")
    print(f"Total bets: {total_bets}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"ROI: {roi:.2%}")


def log_bets(final_bets: pd.DataFrame) -> None:
    file_path = os.path.abspath(TRACKING_FILE)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    now = datetime.now()
    df = final_bets.copy()
    df["date"] = now.strftime("%Y-%m-%d")
    df["bet_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
    df["result"] = "pending"
    df["profit"] = 0.0
    if "closing_odds" not in df.columns:
        df["closing_odds"] = pd.NA
    if "closing_line" not in df.columns:
        df["closing_line"] = pd.NA

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        existing = _ensure_tracking_columns(existing)
        df = pd.concat([existing, df], ignore_index=True)

    df = _ensure_tracking_columns(df)
    df = _recompute_profit(df)
    df.to_csv(file_path, index=False)

    print("\n[TRACKING DEBUG]")
    print("Saved tracking file to:", file_path)
    print("File exists:", os.path.exists(file_path))
    print("Current working directory:", os.getcwd())
    print("Absolute path:", file_path)
    print_performance_summary(df, exported_bets_count=len(final_bets))


def show_performance() -> None:
    file_path = os.path.abspath(TRACKING_FILE)
    if not os.path.exists(file_path):
        print("No tracking file found.")
        return

    df = pd.read_csv(file_path)
    df = _ensure_tracking_columns(df)
    df = _recompute_profit(df)
    print_performance_summary(df)
