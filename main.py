"""Entrypoint for daily cards and backtesting summary outputs."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import logging
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
from sports_betting.sports.common.final_game_filter import filter_predictions_today
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


LOGGER = logging.getLogger(__name__)


def apply_smart_bet_filter(df):

    import pandas as pd

    if df is None or len(df) == 0:
        return df

    selected = []

    for _, row in df.iterrows():

        model_prob = row.get("model_prob", 0)
        market_prob = row.get("market_prob", 0)

        if market_prob == 0:
            continue

        # MARKET-ADJUSTED PROBABILITY (CRITICAL)
        adjusted_prob = 0.75 * market_prob + 0.25 * model_prob
        edge = adjusted_prob - market_prob

        # FILTER RULES

        # 1. Minimum edge
        if edge < 0.025:
            continue

        # 2. Confidence band (removes fake 0.99 probabilities)
        if adjusted_prob < 0.53 or adjusted_prob > 0.68:
            continue

        # 3. Sanity check vs market
        if abs(model_prob - market_prob) > 0.15:
            continue

        row["adjusted_prob"] = adjusted_prob
        row["edge"] = edge

        selected.append(row)

    result = pd.DataFrame(selected)

    if len(result) == 0:
        return result

    return result.sort_values("edge", ascending=False).head(5)


def _normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(name)).split())



def _validate_game_odds_mapping(game_id: str, game_row: dict, game: str) -> tuple[bool, str | None]:
    home_team = str(game_row["home_team"])
    away_team = str(game_row["away_team"])
    home_odds = int(game_row["home_odds"])
    away_odds = int(game_row["away_odds"])

    normalized_home = _normalize_team_name(home_team)
    normalized_away = _normalize_team_name(away_team)
    if not normalized_home or not normalized_away or normalized_home == normalized_away:
        return False, f"invalid normalized teams for {game}: home={home_team} away={away_team}"

    LOGGER.info(
        "[ODDS_MAP] game_id=%s game=%s away_team=%s home_team=%s away_ml=%s home_ml=%s sportsbook_event_away=%s sportsbook_event_home=%s sportsbook_away_outcome=%s sportsbook_home_outcome=%s away_spread=%s@%s home_spread=%s@%s",
        game_id,
        game,
        away_team,
        home_team,
        away_odds,
        home_odds,
        game_row.get("sportsbook_event_away_team", ""),
        game_row.get("sportsbook_event_home_team", ""),
        game_row.get("sportsbook_away_outcome_name", ""),
        game_row.get("sportsbook_home_outcome_name", ""),
        int(game_row["away_spread_odds"]),
        -float(game_row.get("spread_line", 0.0)),
        int(game_row["home_spread_odds"]),
        float(game_row.get("spread_line", 0.0)),
    )
    return True, None


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
    "support_count",
    "composite_score",
    "reason_summary",
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
    """Create candidate bets for a game from model prediction rows."""
    prediction_map: dict[tuple[str, str], dict] = {}
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
            prediction_map[key] = pred

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

        pred_row = prediction_map.get((market, selection), {})
        model_probability = float(pred_row.get("model_probability", 0.5))
        market_probability = 1 / decimal_odds
        edge = model_probability - market_probability
        expected_value = (model_probability * (decimal_odds - 1)) - (1 - model_probability)
        metadata = pred_row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = ast.literal_eval(metadata)
            except (ValueError, SyntaxError):
                metadata = {}
        support_count = int(metadata.get("support_count", 0)) if isinstance(metadata, dict) else 0
        reason_summary = str(pred_row.get("reason_summary", ""))
        confidence = float(pred_row.get("confidence", model_probability))

        # Composite trust score: reward supported edges and de-emphasize pure longshot EV.
        moneyline_penalty = 0.0
        if market == "moneyline":
            moneyline_penalty -= 0.3
            if american_odds >= 180 and support_count < 4:
                moneyline_penalty -= 0.8
        composite_score = (
            1.9 * edge
            + 1.1 * expected_value
            + 0.9 * confidence
            + 0.22 * support_count
            + moneyline_penalty
        )

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
                "confidence": confidence,
                "support_count": support_count,
                "composite_score": composite_score,
                "reason_summary": reason_summary,
            }
        )

    return bets



def _validate_exported_bets_against_sportsbook(game_id: str, game_row: dict, bets: list[dict]) -> tuple[bool, str | None]:
    team_moneyline = {
        _normalize_team_name(str(game_row["home_team"])): int(game_row["home_odds"]),
        _normalize_team_name(str(game_row["away_team"])): int(game_row["away_odds"]),
    }
    for bet in bets:
        if bet["market"] != "moneyline":
            continue
        team_key = _normalize_team_name(str(bet["selection"]))
        sportsbook_odds = team_moneyline.get(team_key)
        if sportsbook_odds is None:
            return False, f"selection team not found in sportsbook mapping: {bet['selection']}"
        if int(bet["odds"]) != sportsbook_odds:
            LOGGER.warning(
                "[ODDS_EXPORT_MISMATCH] game_id=%s selection=%s exported_odds=%s sportsbook_odds=%s home_team=%s away_team=%s sportsbook_event_home=%s sportsbook_event_away=%s",
                game_id,
                bet["selection"],
                int(bet["odds"]),
                sportsbook_odds,
                game_row["home_team"],
                game_row["away_team"],
                game_row.get("sportsbook_event_home_team", ""),
                game_row.get("sportsbook_event_away_team", ""),
            )
            return (
                False,
                f"odds mismatch for selection={bet['selection']} exported={bet['odds']} sportsbook={sportsbook_odds}",
            )
    return True, None

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
                "sportsbook_event_home_team": game_row.get("sportsbook_event_home_team", game_row.get("home_team", "")),
                "sportsbook_event_away_team": game_row.get("sportsbook_event_away_team", game_row.get("away_team", "")),
                "sportsbook_home_outcome_name": game_row.get("sportsbook_home_outcome_name", game_row.get("home_team", "")),
                "sportsbook_away_outcome_name": game_row.get("sportsbook_away_outcome_name", game_row.get("away_team", "")),
            }

    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing outputs to: %s", out_dir.resolve())
    logger.info("Total games processed: %s", total_games_processed)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    predictions_df = filter_predictions_today(predictions_df)
    predictions_df = apply_smart_bet_filter(predictions_df)
    save_dataframe(predictions_df, out_dir / "predictions.csv")

    loaded_predictions_df = pd.read_csv(out_dir / "predictions.csv")
    bets = []
    for game_id, game_predictions in loaded_predictions_df.groupby("game_id"):
        game_context = game_rows_by_id.get(game_id)
        if game_context is None:
            continue
        is_valid, warning = _validate_game_odds_mapping(game_id, game_context, game_context["game"])
        if not is_valid:
            logger.warning("[ODDS_SANITY_SKIP] game_id=%s game=%s reason=%s", game_id, game_context["game"], warning)
            continue
        game_bets = _build_game_candidate_bets(
            predictions=game_predictions.to_dict("records"),
            game_row=game_context,
            sport_name=game_context["sport"],
            game=game_context["game"],
        )
        exports_valid, export_warning = _validate_exported_bets_against_sportsbook(game_id, game_context, game_bets)
        if not exports_valid:
            logger.warning("[ODDS_SANITY_SKIP] game_id=%s game=%s reason=%s", game_id, game_context["game"], export_warning)
            continue
        bets.extend(game_bets)

    filtered_bets = []
    for bet in bets:
        if bet["support_count"] < 2:
            continue
        if bet["market"] == "moneyline" and abs(int(bet["odds"])) >= 180 and bet["support_count"] < 4:
            continue
        if bet["market"] == "moneyline" and bet["model_probability"] < 0.46:
            continue
        filtered_bets.append(bet)

    ranked_bets = sorted(filtered_bets, key=lambda x: (x["composite_score"], x["confidence"]), reverse=True)
    trimmed_recs = ranked_bets[:TOP_BETS_DAILY] if not loaded_predictions_df.empty else []
    logger.info("Total bets exported: %s", len(trimmed_recs))

    recommendations_df = pd.DataFrame(trimmed_recs)
    recommendations_df = recommendations_df.reindex(columns=RECOMMENDATION_COLUMNS)
    recommendations_df = filter_predictions_today(recommendations_df)
    recommendations_df = apply_smart_bet_filter(recommendations_df)
    trimmed_recs = recommendations_df.to_dict("records")
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
