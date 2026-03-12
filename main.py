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
)
from sports_betting.sports.common.logging_utils import setup_logging
from sports_betting.sports.common.reporting import render_daily_card, save_dataframe, save_recommendations_json
from sports_betting.sports.common.risk import BankrollConfig
from sports_betting.sports.common.selection import qualify_prediction
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
    "metadata",
]

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
]


def choose_model(sport: str):
    return {"nba": NBAModel, "nfl": NFLModel, "nhl": NHLModel}[sport]()


def run_daily_pipeline(config_path: str | None = None, sport: str | None = None) -> None:
    cfg = load_config(config_path)
    logger = setup_logging(cfg["logging"]["level"])

    all_predictions: list[dict] = []
    all_recs = []
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

    total_games_processed = 0

    for sport_name in sports_to_run:
        logger.info("Running %s pipeline", sport_name)
        model = choose_model(sport_name)
        historical, daily = load_historical_and_daily(sport_name)
        total_games_processed += len(daily)

        artifact_path = model_artifact_path(sport_name)
        if live_mode and historical.empty:
            model.load_artifact(artifact_path)
            logger.info("Loaded trained %s model artifact: %s", sport_name.upper(), artifact_path)
        else:
            model.train(historical)
            if live_mode:
                model.save_artifact(artifact_path)
                logger.info("Trained and saved %s model artifact: %s", sport_name.upper(), artifact_path)

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
            rec = qualify_prediction(
                pred=pred,
                game_text=game,
                event_date=pd.to_datetime(game_row["event_date"]).date(),
                odds=int(game_row[odds_col]),
                line=float(game_row[line_col]) if line_col else None,
                thresholds=cfg["thresholds"][sport_name][pred.market],
                bankroll_cfg=bankroll_cfg,
                stake_mode=cfg["bankroll"]["stake_mode"],
            )
            if rec:
                all_recs.append(rec)
                recs_by_sport[sport_name].append(rec)
            else:
                all_passes.append(
                    f"{game} ({sport_name.upper()} {pred.market} {pred.side}): filtered by edge/EV/confidence."
                )

    all_recs.sort(key=lambda x: x.rank_score, reverse=True)
    top_n = cfg["output"]["top_n"]
    trimmed_recs = all_recs[:top_n] if top_n > 0 else all_recs

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
        f"Total recommendations qualified: {len(all_recs)}",
        f"Total recommendations exported (after top_n): {len(trimmed_recs)}",
        "",
    ]
    if trimmed_recs:
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
