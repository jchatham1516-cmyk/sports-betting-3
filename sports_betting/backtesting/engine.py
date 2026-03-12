"""Walk-forward style backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    bets: int
    wins: int
    losses: int
    pushes: int
    units: float
    roi: float
    hit_rate: float
    avg_edge: float
    max_drawdown: float


def compute_drawdown(equity_curve: list[float]) -> float:
    peak = -np.inf
    drawdowns = []
    for x in equity_curve:
        peak = max(peak, x)
        drawdowns.append(peak - x)
    return float(max(drawdowns)) if drawdowns else 0.0


def summarize_backtest(results: pd.DataFrame) -> BacktestSummary:
    bets = len(results)
    wins = int((results["result_units"] > 0).sum())
    losses = int((results["result_units"] < 0).sum())
    pushes = int((results["result_units"] == 0).sum())
    units = float(results["result_units"].sum())
    staked = float(results["stake_units"].sum()) if bets else 0.0
    roi = (units / staked) if staked else 0.0
    hit_rate = wins / bets if bets else 0.0
    avg_edge = float(results["edge"].mean()) if bets else 0.0
    eq = results["result_units"].cumsum().tolist()
    dd = compute_drawdown(eq)
    return BacktestSummary(bets, wins, losses, pushes, units, roi, hit_rate, avg_edge, dd)
