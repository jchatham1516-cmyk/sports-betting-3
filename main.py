"""Entrypoint for daily cards and backtesting summary outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from sports_betting.backtesting.engine import summarize_backtest
from sports_betting.config.settings import load_config
from sports_betting.scripts.data_io import (
    is_test_mode,
    load_historical_and_daily,
    model_artifact_path,
    validate_historical_requirements,
    validate_model_artifacts_exist,
)
from sports_betting.sports.common.logging_utils import setup_logging
from sports_betting.sports.common.reporting import render_daily_card, save_dataframe, save_recommendations_json
from sports_betting.sports.common.risk import BankrollConfig
from sports_betting.sports.common.selection import candidate_prediction
from sports_betting.sports.nba.model import NBAModel
from sports_betting.sports.nfl.model import NFLModel
from sports_betting.sports.nhl.model import NHLModel


PREDICTION_COLUMNS = [
    "game_id",
    "sport",
    "market",
    "side",
    "model_probability",
    "market_implied_probability",
    "edge",
    "expected_value",
    "confidence",
    "reason_summary",
    "flags",
    "market_prob",
    "metadata",
]

MIN_EDGE = 0.01
MIN_EV = -0.005
MIN_CONFIDENCE = 0.55
MAX_BETS = 8

RECOMMENDATION_COLUMNS = [
    "event_date",
    "sport",
    "game",
    "market",
    "side",
    "line",
    "odds",
    "model_probability",
    "market_implied_probability",
    "edge",
    "expected_value",
    "confidence_score",
    "confidence_tier",
    "recommended_units",
    "reason_summary",
    "flags",
    "rank_score",
    "market_prob",
    "model_prob",
    "line_movement",
    "clv_placeholder",
    "injury_confidence_score",
    "opening_line",
    "bet_line",
    "current_line",
    "closing_line",
    "clv_diff",
    "fallback_pick",
]


def choose_model(sport: str):
    return {"nba": NBAModel, "nfl": NFLModel, "nhl": NHLModel}[sport]()


def _best_market_sides(bets: list) -> list:
    """Keep only one side per sport/game/market using highest EV/edge/confidence."""
    best = {}
    for bet in bets:
        key = (bet.sport, bet.game, bet.market)
        current = best.get(key)
        if current is None or (
            bet.expected_value,
            bet.edge,
            bet.confidence_score,
            bet.rank_score,
        ) > (
            current.expected_value,
            current.edge,
            current.confidence_score,
            current.rank_score,
        ):
            best[key] = bet
    return list(best.values())


def _bet_metrics(bet) -> dict[str, float]:
    return {
        "edge": bet.edge,
        "ev": bet.expected_value,
        "confidence": bet.confidence_score,
    }


def run_daily_pipeline(config_path: str | None = None, sport: str | None = None) -> None:
    cfg = load_config(config_path)
    logger = setup_logging(cfg["logging"]["level"])

    all_predictions: list[dict] = []
    candidate_pool = []
    all_passes: list[str] = []
    recs_by_sport = defaultdict(list)

    bankroll_cfg = BankrollConfig(
        bankroll=cfg["bankroll"]["bankroll"],
        unit_size=cfg["bankroll"]["unit_size"],
        max_units_per_bet=cfg["bankroll"]["max_units_per_bet"],
        kelly_fraction=cfg["bankroll"]["kelly_fraction"],
    )

    sports_to_run = [sport] if sport else cfg["sports_enabled"]

    logger.info("Sports selected for this run: %s", ", ".join(sports_to_run))

    live_mode = not is_test_mode()
    if live_mode:
        validate_historical_requirements(sports=sports_to_run, allow_model_artifacts=True, validate_schema=True)
        validate_model_artifacts_exist(sports=sports_to_run)

    total_games_processed = 0

    for sport_name in sports_to_run:
        logger.info("Running %s pipeline", sport_name)
        model = choose_model(sport_name)
        historical, daily = load_historical_and_daily(sport_name)
        total_games_processed += len(daily)

        artifact_path = model_artifact_path(sport_name)
        if live_mode:
            model.load_artifact(artifact_path)
            logger.info("Loaded trained %s model artifact: %s", sport_name.upper(), artifact_path)
        else:
            model.train(historical)

        preds = model.predict_daily(daily)
        logger.info("%s games processed for %s (%s predictions generated)", len(daily), sport_name, len(preds))

        for pred in preds:
            all_predictions.append(asdict(pred))
            game = pred.metadata.get("game", pred.game_id)
            game_row = daily[daily["game_id"] == pred.game_id].iloc[0]
            odds_col = {
                "moneyline": "home_odds" if pred.side == game_row["home_team"] else "away_odds",
                "spread": "home_spread_odds" if pred.side.startswith(game_row["home_team"]) else "away_spread_odds",
                "total": "over_odds" if pred.side.startswith("Over") else "under_odds",
            }[pred.market]
            line_col = {"moneyline": None, "spread": "spread_line", "total": "total_line"}[pred.market]
            rec = candidate_prediction(
                pred=pred,
                game_text=game,
                event_date=pd.to_datetime(game_row["event_date"]).date(),
                odds=int(game_row[odds_col]),
                line=float(game_row[line_col]) if line_col else None,
                bankroll_cfg=bankroll_cfg,
                stake_mode=cfg["bankroll"]["stake_mode"],
            )
            if rec:
                candidate_pool.append(rec)

    # Remove duplicate sides first: keep one bet per game + market.
    bets = _best_market_sides(candidate_pool)

    qualified = [
        b
        for b in bets
        if _bet_metrics(b).get("edge", 0) >= MIN_EDGE
        and _bet_metrics(b).get("ev", 0) >= MIN_EV
        and _bet_metrics(b).get("confidence", 0) >= MIN_CONFIDENCE
    ]

    filtered_out = len(bets) - len(qualified)
    if filtered_out > 0:
        qualified_ids = {id(q) for q in qualified}
        for b in bets:
            if id(b) in qualified_ids:
                continue
            all_passes.append(
                f"{b.game} ({b.sport.upper()} {b.market} {b.side}): filtered by edge/EV/confidence."
            )

    used_fallback = False
    if not qualified:
        fallback = sorted(
            bets,
            key=lambda x: (x.expected_value, x.edge, x.confidence_score),
            reverse=True,
        )

        qualified = fallback[:5]

        for b in qualified:
            b.fallback_pick = True

        used_fallback = bool(qualified)

    # Ensure at least 3-5 bets whenever games exist.
    if total_games_processed > 0 and len(qualified) < 3:
        selected = {
            (b.sport, b.game, b.market, b.side, b.odds, b.line)
            for b in qualified
        }
        remainder = sorted(
            bets,
            key=lambda x: (x.expected_value, x.edge, x.confidence_score),
            reverse=True,
        )
        for b in remainder:
            rec_key = (b.sport, b.game, b.market, b.side, b.odds, b.line)
            if rec_key in selected:
                continue
            b.fallback_pick = True
            qualified.append(b)
            selected.add(rec_key)
            if len(qualified) >= 3:
                break

    qualified.sort(key=lambda x: (x.rank_score, x.expected_value), reverse=True)
    top_n = min(int(cfg["output"].get("top_n", MAX_BETS)), MAX_BETS)
    if total_games_processed > 0:
        top_n = max(top_n, 3)
    trimmed_recs = qualified[:top_n] if top_n > 0 else qualified

    recs_by_sport = defaultdict(list)
    for rec in trimmed_recs:
        recs_by_sport[rec.sport].append(rec)

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing outputs to: %s", out_dir.resolve())
    logger.info("Total games processed: %s", total_games_processed)
    logger.info("Total qualified bets: %s", len(trimmed_recs))

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    save_dataframe(predictions_df, out_dir / "predictions.csv")

    recommendations_df = pd.DataFrame([asdict(r) for r in trimmed_recs])
    recommendations_df = recommendations_df.reindex(columns=RECOMMENDATION_COLUMNS)
    save_dataframe(recommendations_df, out_dir / "recommended_bets.csv")

    save_recommendations_json(trimmed_recs, out_dir / "daily_recommendations.json")

    # simple synthetic backtest summary using recommendation EV as proxy; replace with realized grading data in production
    if trimmed_recs:
        bt = pd.DataFrame([asdict(r) for r in trimmed_recs])
        bt["result_units"] = bt["expected_value"] * bt["recommended_units"]
        bt["stake_units"] = bt["recommended_units"]
        bt_summary = summarize_backtest(bt)
        save_dataframe(pd.DataFrame([asdict(bt_summary)]), out_dir / "backtest_summary.csv")

    card = render_daily_card(datetime.utcnow().strftime("%Y-%m-%d"), recs_by_sport, all_passes[:20])
    print(card)
    report_lines = [
        f"Output directory: {out_dir.resolve()}",
        f"Games processed: {total_games_processed}",
        f"Total recommendations qualified: {len(qualified)}",
        f"Total recommendations exported (after top_n): {len(trimmed_recs)}",
        "",
    ]
    if trimmed_recs and used_fallback:
        report_lines.append("No bets met thresholds — exporting fallback picks.")
    elif trimmed_recs:
        report_lines.append("Qualifying bets were found.")
    else:
        report_lines.append("No qualifying bets today")

    report_lines.extend(["", card])
    (out_dir / "daily_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sports betting daily pipeline.")
    parser.add_argument("--config", default=None, help="Optional path to a custom config file.")
    parser.add_argument("--sport", choices=["nba", "nfl", "nhl"], default=None, help="Run only one sport.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_daily_pipeline(config_path=args.config, sport=args.sport)
