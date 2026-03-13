"""Entrypoint for daily cards and backtesting summary outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
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
from sports_betting.sports.common.reporting import save_dataframe
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

TOP_BETS_DAILY = 5

RECOMMENDATION_COLUMNS = [
    "sport",
    "game",
    "market",
    "selection",
    "odds",
    "model_probability",
    "market_probability",
    "edge",
    "expected_value",
    "confidence",
]


def choose_model(sport: str):
    return {"nba": NBAModel, "nfl": NFLModel, "nhl": NHLModel}[sport]()


def _american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return 1 + (odds / 100.0)
    if odds <= -100:
        return 1 + (100.0 / abs(odds))
    return 0.0


def _build_game_candidate_bets(predictions: list[dict], game_row: dict, sport_name: str, game: str) -> list[dict]:
    """Create all six candidate bet types for a game from available prediction rows."""
    prediction_map: dict[tuple[str, str], float] = {}
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])
    total_line = float(game_row.get("total_line", 0.0))

    for pred in predictions:
        market = "totals" if pred["market"] in {"total", "totals"} else pred["market"]
        side = str(pred["side"])
        key = None
        if market == "moneyline":
            key = (market, "moneyline_home" if side == home_team else "moneyline_away")
        elif market == "spread":
            key = (market, "spread_home" if side.startswith(home_team) else "spread_away")
        elif market == "totals":
            key = (market, "over" if side.startswith("Over") else "under")
        if key:
            prediction_map[key] = float(pred["model_probability"])

    candidate_specs = [
        ("moneyline", "moneyline_home", int(game_row["home_odds"])),
        ("moneyline", "moneyline_away", int(game_row["away_odds"])),
        ("spread", "spread_home", int(game_row["home_spread_odds"])),
        ("spread", "spread_away", int(game_row["away_spread_odds"])),
        ("totals", "over", int(game_row["over_odds"])),
        ("totals", "under", int(game_row["under_odds"])),
    ]

    bets: list[dict] = []
    for market, selection, american_odds in candidate_specs:
        decimal_odds = _american_to_decimal(american_odds)
        if decimal_odds <= 1:
            continue

        model_probability = prediction_map.get((market, selection), 0.5)
        market_probability = 1 / decimal_odds
        edge = model_probability - market_probability
        expected_value = (model_probability * (decimal_odds - 1)) - (1 - model_probability)
        if market == "spread":
            selection_label = home_team if selection == "spread_home" else away_team
        elif market == "moneyline":
            selection_label = home_team if selection == "moneyline_home" else away_team
        else:
            selection_label = f"Over {total_line:.1f}" if selection == "over" else f"Under {total_line:.1f}"

        bets.append(
            {
                "sport": sport_name,
                "game": game,
                "market": market,
                "selection": selection_label,
                "odds": american_odds,
                "model_probability": model_probability,
                "market_probability": market_probability,
                "edge": edge,
                "expected_value": expected_value,
                "confidence": model_probability,
            }
        )

    return bets


def run_daily_pipeline(config_path: str | None = None, sport: str | None = None) -> None:
    cfg = load_config(config_path)
    logger = setup_logging(cfg["logging"]["level"])

    all_predictions: list[dict] = []
    game_rows_by_id: dict[str, dict] = {}

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

        predictions_by_game_id: dict[str, list[dict]] = defaultdict(list)
        for pred in preds:
            all_predictions.append(asdict(pred))
            game_row = daily[daily["game_id"] == pred.game_id].iloc[0]
            predictions_by_game_id[pred.game_id].append(asdict(pred))
            game_rows_by_id[pred.game_id] = {
                "home_team": game_row["home_team"],
                "away_team": game_row["away_team"],
                "home_odds": game_row["home_odds"],
                "away_odds": game_row["away_odds"],
                "home_spread_odds": game_row["home_spread_odds"],
                "away_spread_odds": game_row["away_spread_odds"],
                "over_odds": game_row["over_odds"],
                "under_odds": game_row["under_odds"],
                "total_line": game_row.get("total_line", 0.0),
                "game": pred.metadata.get("game", pred.game_id),
                "sport": sport_name,
            }

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing outputs to: %s", out_dir.resolve())
    logger.info("Total games processed: %s", total_games_processed)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    save_dataframe(predictions_df, out_dir / "predictions.csv")

    loaded_predictions_df = pd.read_csv(out_dir / "predictions.csv")
    bets = []
    for game_id, game_predictions in loaded_predictions_df.groupby("game_id"):
        game_context = game_rows_by_id.get(game_id)
        if game_context is None:
            continue
        bets.extend(
            _build_game_candidate_bets(
                predictions=game_predictions.to_dict("records"),
                game_row=game_context,
                sport_name=game_context["sport"],
                game=game_context["game"],
            )
        )

    ranked_bets = sorted(
        bets,
        key=lambda x: (x["expected_value"], x["edge"], x["confidence"]),
        reverse=True,
    )
    trimmed_recs = ranked_bets[:TOP_BETS_DAILY] if not loaded_predictions_df.empty else []
    logger.info("Total bets exported: %s", len(trimmed_recs))

    recommendations_df = pd.DataFrame(trimmed_recs)
    recommendations_df = recommendations_df.reindex(columns=RECOMMENDATION_COLUMNS)
    save_dataframe(recommendations_df, out_dir / "recommended_bets.csv")

    if trimmed_recs:
        bt = recommendations_df.copy()
        bt["recommended_units"] = 1.0
        bt["result_units"] = bt["expected_value"] * bt["recommended_units"]
        bt["stake_units"] = bt["recommended_units"]
        bt_summary = summarize_backtest(bt)
        save_dataframe(pd.DataFrame([asdict(bt_summary)]), out_dir / "backtest_summary.csv")

    card = "Top model bets exported from predictions." if trimmed_recs else "No bets exported today"
    print(card)
    report_lines = [
        f"Output directory: {out_dir.resolve()}",
        f"Games processed: {total_games_processed}",
        f"Total ranked recommendations available: {len(ranked_bets)}",
        f"Total recommendations exported (top {TOP_BETS_DAILY}): {len(trimmed_recs)}",
        "",
    ]
    if trimmed_recs:
        report_lines.append("Top model bets exported from predictions.")
    else:
        report_lines.append("No bets exported today")

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
