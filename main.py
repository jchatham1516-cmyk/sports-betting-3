"""Entrypoint for daily cards and backtesting summary outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from sports_betting.backtesting.engine import summarize_backtest
from sports_betting.config.settings import load_config
from sports_betting.scripts.data_io import load_historical_and_daily
from sports_betting.sports.common.logging_utils import setup_logging
from sports_betting.sports.common.reporting import render_daily_card, save_dataframe, save_recommendations_json
from sports_betting.sports.common.risk import BankrollConfig
from sports_betting.sports.common.selection import qualify_prediction
from sports_betting.sports.nba.model import NBAModel
from sports_betting.sports.nfl.model import NFLModel
from sports_betting.sports.nhl.model import NHLModel


def choose_model(sport: str):
    return {"nba": NBAModel, "nfl": NFLModel, "nhl": NHLModel}[sport]()


def run_daily_pipeline(config_path: str | None = None) -> None:
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

    for sport in cfg["sports_enabled"]:
        logger.info("Running %s pipeline", sport)
        model = choose_model(sport)
        historical, daily = load_historical_and_daily(sport)
        model.train(historical)
        preds = model.predict_daily(daily)

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
                thresholds=cfg["thresholds"][sport][pred.market],
                bankroll_cfg=bankroll_cfg,
                stake_mode=cfg["bankroll"]["stake_mode"],
            )
            if rec:
                all_recs.append(rec)
                recs_by_sport[sport].append(rec)
            else:
                all_passes.append(f"{game} ({sport.upper()} {pred.market} {pred.side}): filtered by edge/EV/confidence.")

    all_recs.sort(key=lambda x: x.rank_score, reverse=True)
    top_n = cfg["output"]["top_n"]
    trimmed_recs = all_recs[:top_n] if top_n > 0 else all_recs

    out_dir = Path("sports_betting/data/outputs")
    save_dataframe(pd.DataFrame(all_predictions), out_dir / "daily_predictions.csv")
    save_recommendations_json(trimmed_recs, out_dir / "daily_recommendations.json")
    save_dataframe(pd.DataFrame([asdict(r) for r in trimmed_recs]), out_dir / "daily_recommendations.csv")

    # simple synthetic backtest summary using recommendation EV as proxy; replace with realized grading data in production
    if trimmed_recs:
        bt = pd.DataFrame([asdict(r) for r in trimmed_recs])
        bt["result_units"] = bt["expected_value"] * bt["recommended_units"]
        bt["stake_units"] = bt["recommended_units"]
        bt_summary = summarize_backtest(bt)
        save_dataframe(pd.DataFrame([asdict(bt_summary)]), out_dir / "backtest_summary.csv")

    card = render_daily_card(datetime.utcnow().strftime("%Y-%m-%d"), recs_by_sport, all_passes[:20])
    print(card)
    (out_dir / "daily_card.txt").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    run_daily_pipeline()
