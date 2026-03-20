"""Entrypoint for daily cards and backtesting summary outputs."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sports_betting.backtesting.engine import summarize_backtest
from sports_betting.config.settings import load_config
from sports_betting.models.entities import Prediction
from sports_betting.scripts.data_io import (
    historical_file_path,
    is_test_mode,
    load_historical_and_daily,
    model_artifact_path,
    validate_historical_requirements,
)
from sports_betting.sports.common.final_game_filter import filter_predictions_today
from sports_betting.sports.common.logging_utils import setup_logging
from sports_betting.sports.common.odds import expected_value
from sports_betting.sports.common.reporting import save_dataframe
from sports_betting.sports.nba.model import NBAModel
from sports_betting.sports.nba.simple_model import american_to_implied_prob
from sports_betting.sports.nba.simple_model import FEATURE_COLUMNS as NBA_RUNTIME_FEATURE_COLUMNS
from sports_betting.sports.nba.simple_model import train_runtime_model
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
EDGE_THRESHOLD = 0.01
FORCE_BETS = True
LOGGER = logging.getLogger(__name__)


def estimate_elo_from_moneyline(ml):
    if ml < 0:
        return 1500 + min(abs(ml), 400) * 0.5
    else:
        return 1500 - min(ml, 400) * 0.5


def generate_sharp_edge(row):
    ml = row.get("moneyline", 0)
    spread = row.get("spread", 0)

    # Convert market probability
    if "market_prob" not in row or row["market_prob"] == 0:
        if ml < 0:
            market_prob = abs(ml) / (abs(ml) + 100)
        else:
            market_prob = 100 / (ml + 100)
    else:
        market_prob = row["market_prob"]

    # Moneyline strength scaling
    if ml < 0:
        ml_strength = min(abs(ml) / 400, 1)
    else:
        ml_strength = -min(ml / 400, 1)

    # Spread strength scaling
    spread_strength = min(abs(spread) / 12, 1)

    # Favorite bias (favorites win more often)
    favorite_boost = 0.03 if ml < -150 else 0

    # Underdog value bias
    dog_boost = 0.02 if 120 < ml < 250 else 0

    # Build final probability
    final_prob = (
        0.65 * market_prob
        + 0.20 * (0.5 + ml_strength * 0.25)
        + 0.15 * (0.5 + spread_strength * 0.2)
        + favorite_boost
        + dog_boost
    )

    # Clamp probability
    final_prob = max(0.05, min(0.95, final_prob))

    return final_prob


def apply_smart_bet_filter(df):

    import pandas as pd

    if df is None or len(df) == 0:
        return df

    results = []

    for _, row in df.iterrows():
        home_ml = row.get("home_moneyline", row.get("home_odds", 0))
        away_ml = row.get("away_moneyline", row.get("away_odds", 0))

        elo_home = estimate_elo_from_moneyline(home_ml)
        elo_away = estimate_elo_from_moneyline(away_ml)

        row["elo_diff"] = elo_home - elo_away

        # stronger teams usually have bigger spreads
        spread = row.get("spread", 0)
        row["net_rating_diff"] = -spread * 1.5
        row["form_diff"] = -spread * 0.8

        # REST PROXY (based on spread = stronger teams handle fatigue better)
        spread = abs(row.get("spread", 0))
        if spread >= 8:
            row["rest_diff"] = 1
        elif spread <= 3:
            row["rest_diff"] = -1
        else:
            row["rest_diff"] = 0

        # TRAVEL FATIGUE (underdogs more affected)
        ml = row.get("moneyline", 0)
        if ml > 150:
            row["travel_fatigue_diff"] = 0.5
        elif ml < -200:
            row["travel_fatigue_diff"] = -0.5
        else:
            row["travel_fatigue_diff"] = 0

        # better structured injury proxy
        if ml < -300:
            row["injury_impact_diff"] = -0.3  # favorites overpriced risk
        elif 120 < ml < 250:
            row["injury_impact_diff"] = 0.2   # dogs undervalued
        else:
            row["injury_impact_diff"] = 0

        spread = row.get("spread", 0)
        if abs(spread) >= 10:
            row["predictability"] = 1
        elif abs(spread) <= 3:
            row["predictability"] = -1
        else:
            row["predictability"] = 0

        if ml < 0:
            ml_strength = abs(ml) / 100
        else:
            ml_strength = -ml / 100
        row["line_disagreement"] = ml_strength - spread

        if ml < -300:
            row["public_bias"] = -1
        elif ml > 200:
            row["public_bias"] = 1
        else:
            row["public_bias"] = 0

        model_prob = row.get("model_prob", 0)
        market_prob = row.get("market_prob", 0)

        row["model_prob"] = generate_sharp_edge(row)
        # normalize elo_diff impact
        elo_effect = row["elo_diff"] / 150
        existing_feature_boost = (
            0.03 * elo_effect
            + 0.02 * (row["net_rating_diff"] / 10)
            + 0.015 * row["rest_diff"]
            - 0.015 * row["travel_fatigue_diff"]
            + 0.01 * row["injury_impact_diff"]
        )
        feature_boost = (
            0.025 * (row["form_diff"] / 10)
            + 0.02 * row["predictability"]
            + 0.02 * (row["line_disagreement"] / 10)
            + 0.015 * row["public_bias"]
        )
        row["model_prob"] = row["model_prob"] + existing_feature_boost + feature_boost
        row["model_prob"] = max(0.05, min(0.95, row["model_prob"]))
        model_prob = row.get("model_prob", model_prob)

        if market_prob == 0:
            continue

        # Blend model + market (very important)
        adjusted_prob = 0.7 * market_prob + 0.3 * model_prob
        edge = (model_prob - market_prob) * 1.5

        row["adjusted_prob"] = adjusted_prob
        row["edge"] = edge

        results.append(row)

    df = pd.DataFrame(results)

    if len(df) == 0:
        return df

    sort_key = "expected_value" if "expected_value" in df.columns else "edge"
    df_sorted = df.sort_values(by=sort_key, ascending=False)

    print("[DEBUG] Top edges:")
    debug_cols = ["away_team", "home_team", "edge", "expected_value"]
    available_debug_cols = [col for col in debug_cols if col in df_sorted.columns]
    print(df_sorted[available_debug_cols].head(10))

    if FORCE_BETS:
        return df_sorted.head(3)

    # PRIMARY FILTER (normal mode)
    filtered = df[
        (df["edge"] > EDGE_THRESHOLD) &
        (df["expected_value"] > 0) &
        (df["adjusted_prob"].between(0.50, 0.75))
    ]

    # 🔥 FALLBACK MODE (CRITICAL)
    if len(filtered) == 0:
        print("No strong bets → using fallback mode")
        return df_sorted.head(5)

    return filtered.sort_values(by=sort_key, ascending=False).head(5)


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


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def get_payout(odds: int) -> float:
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def _first_existing_column(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.full(len(frame), default), index=frame.index, dtype=float)


def train_runtime_home_win_model(historical_df: pd.DataFrame, sport_name: str):
    try:
        assert "home_moneyline" in historical_df.columns
        assert "spread" in historical_df.columns
        assert "home_win" in historical_df.columns
    except AssertionError:
        print(
            f"[{sport_name.upper()}] Missing required training columns. "
            f"Available columns: {list(historical_df.columns)}"
        )
        return None

    y = pd.to_numeric(historical_df.get("home_win"), errors="coerce").fillna(0).astype(int)
    print(f"[{sport_name.upper()}] Attempting runtime model training...")
    print(f"[{sport_name.upper()}] Training rows: {len(historical_df)}")

    if len(historical_df) < 10:
        print(f"[{sport_name.upper()}] WARNING: Very small dataset")
        print(f"[{sport_name.upper()}] Runtime home-win model skipped (needs >=10 rows).")
        return None

    if y.nunique() < 2:
        print(f"[{sport_name.upper()}] Runtime home-win model skipped (target requires 2 classes).")
        return None

    model = train_runtime_model(historical_df)
    if model is None:
        print(f"[{sport_name.upper()}] Runtime home-win model skipped (insufficient signal).")
        return None
    print(f"[{sport_name.upper()}] Model trained successfully")
    return model


def predict_runtime(model, games_df: pd.DataFrame):
    df = games_df.copy()
    df_full = df.copy()
    if "home_moneyline" not in df.columns:
        if "home_ml" in df.columns:
            df["home_moneyline"] = df["home_ml"]
        elif "moneyline_home" in df.columns:
            df["home_moneyline"] = df["moneyline_home"]
        elif "odds_home" in df.columns:
            df["home_moneyline"] = df["odds_home"]
        elif "home_odds" in df.columns:
            df["home_moneyline"] = df["home_odds"]

    if "home_moneyline" not in df.columns:
        raise ValueError("home_moneyline is required for runtime prediction")

    if "spread" not in df.columns:
        if "home_spread" in df.columns:
            df["spread"] = df["home_spread"]
        elif "spread_home" in df.columns:
            df["spread"] = df["spread_home"]
        elif "spread_line" in df.columns:
            df["spread"] = df["spread_line"]
        else:
            df["spread"] = 0.0

    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="coerce").fillna(0)
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    if "implied_home_prob" not in df.columns:
        df["implied_home_prob"] = df["home_moneyline"].apply(american_to_implied_prob)
    else:
        df["implied_home_prob"] = pd.to_numeric(df["implied_home_prob"], errors="coerce").fillna(0)
    df["spread_abs"] = df["spread"].abs()
    df["is_favorite"] = (df["home_moneyline"] < 0).astype(int)
    df["spread_value_signal"] = df["spread"] * df["implied_home_prob"]
    print("[NBA] Using real odds for prediction")

    runtime_model = model
    scaler = None
    if isinstance(model, tuple) and len(model) >= 2:
        runtime_model, scaler = model

    feature_columns = getattr(runtime_model, "feature_columns", NBA_RUNTIME_FEATURE_COLUMNS)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features at prediction time: {missing}")

    X = df[feature_columns].copy()

    X = X.fillna(0.0)
    if "implied_home_prob" in X.columns:
        X["implied_home_prob"] = X["implied_home_prob"].replace(0.0, 0.5)

    print(f"[NBA] Prediction columns: {list(df.columns)}")
    if scaler is not None:
        X = scaler.transform(X)

    if hasattr(runtime_model, "feature_columns") and runtime_model.feature_columns is not None:
        X_pred = X
    elif hasattr(X, "values"):
        X_pred = X.values
    else:
        X_pred = X

    probs = runtime_model.predict_proba(X_pred)[:, 1]
    model_probs = probs

    # Keep full, non-feature columns intact for downstream bet readability/debugging.
    df = df_full
    df["model_prob"] = model_probs
    df["edge"] = 0.0
    df["expected_value"] = 0.0
    return model_probs


def _ensure_runtime_prediction_columns(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    if "home_moneyline" not in out.columns:
        if "home_odds" in out.columns:
            out["home_moneyline"] = out["home_odds"]

    if "spread" not in out.columns:
        if "spread_line" in out.columns:
            out["spread"] = out["spread_line"]

    if "home_moneyline" not in out.columns:
        if "home_ml" in out.columns:
            out["home_moneyline"] = out["home_ml"]
        elif "moneyline_home" in out.columns:
            out["home_moneyline"] = out["moneyline_home"]
        elif "odds_home" in out.columns:
            out["home_moneyline"] = out["odds_home"]

    if "spread" not in out.columns:
        if "home_spread" in out.columns:
            out["spread"] = out["home_spread"]
        elif "spread_home" in out.columns:
            out["spread"] = out["spread_home"]
        else:
            out["spread"] = 0.0

    if "home_moneyline" in out.columns:
        out["home_moneyline"] = pd.to_numeric(out["home_moneyline"], errors="coerce").fillna(0)
    if "spread" in out.columns:
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(0)
    if "implied_home_prob" not in out.columns and "home_moneyline" in out.columns:
        out["implied_home_prob"] = out["home_moneyline"].apply(american_to_implied_prob)
    if "spread" in out.columns:
        out["spread_abs"] = out["spread"].abs()
    if "home_moneyline" in out.columns:
        out["is_favorite"] = (out["home_moneyline"] < 0).astype(int)
    return out


def _build_runtime_moneyline_predictions(
    daily_df: pd.DataFrame,
    runtime_model,
    sport_name: str,
) -> list[Prediction]:
    if daily_df.empty or runtime_model is None:
        return []

    runtime_df = daily_df.copy()
    runtime_df["predicted_home_win_prob"] = predict_runtime(runtime_model, runtime_df)
    def _calibrate_prob(p):
        return min(max(p, 0.05), 0.95)
    runtime_df["predicted_home_win_prob"] = runtime_df["predicted_home_win_prob"].apply(_calibrate_prob)
    if "model_prob" not in runtime_df.columns:
        runtime_df["model_prob"] = runtime_df["predicted_home_win_prob"]

    preds: list[Prediction] = []
    for _, row in runtime_df.iterrows():
        game_id = str(row.get("game_id"))
        home_team = str(row.get("home_team"))
        away_team = str(row.get("away_team"))
        game_txt = f"{away_team} @ {home_team}"

        home_prob = float(np.clip(row["predicted_home_win_prob"], 0.01, 0.99))
        away_prob = float(np.clip(1.0 - home_prob, 0.01, 0.99))
        home_market_prob = american_to_implied_prob(int(row["home_odds"]))
        away_market_prob = american_to_implied_prob(int(row["away_odds"]))

        preds.extend(
            [
                Prediction(
                    game_id=game_id,
                    sport=sport_name,
                    market="moneyline",
                    side=home_team,
                    model_probability=home_prob,
                    market_implied_probability=home_market_prob,
                    edge=home_prob - home_market_prob,
                    expected_value=expected_value(home_prob, int(row["home_odds"])),
                    confidence=home_prob,
                    reason_summary="Runtime logistic moneyline prediction",
                    flags=[],
                    market_prob=home_market_prob,
                    metadata={"game": game_txt, "predicted_home_win_prob": home_prob, "model_prob": home_prob},
                ),
                Prediction(
                    game_id=game_id,
                    sport=sport_name,
                    market="moneyline",
                    side=away_team,
                    model_probability=away_prob,
                    market_implied_probability=away_market_prob,
                    edge=away_prob - away_market_prob,
                    expected_value=expected_value(away_prob, int(row["away_odds"])),
                    confidence=away_prob,
                    reason_summary="Runtime logistic moneyline prediction",
                    flags=[],
                    market_prob=away_market_prob,
                    metadata={"game": game_txt, "predicted_home_win_prob": home_prob, "model_prob": away_prob},
                ),
            ]
        )
    return preds


def _apply_runtime_home_win_probabilities(
    preds: list,
    daily_df: pd.DataFrame,
    runtime_model,
) -> list:
    if daily_df.empty:
        return preds

    if runtime_model is not None:
        home_probs = pd.Series(predict_runtime(runtime_model, daily_df), index=daily_df["game_id"].astype(str))
    else:
        implied = pd.to_numeric(_first_existing_column(daily_df, ["home_odds", "home_moneyline"], default=0.0), errors="coerce")
        home_probs = implied.apply(
            lambda odds: abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100) if odds > 0 else 0.5
        )
        home_probs.index = daily_df["game_id"].astype(str)

    game_rows = daily_df.set_index(daily_df["game_id"].astype(str))

    for pred in preds:
        if pred.market != "moneyline":
            continue
        game_id = str(pred.game_id)
        if game_id not in home_probs.index or game_id not in game_rows.index:
            continue
        game_row = game_rows.loc[game_id]
        home_prob = float(np.clip(home_probs.loc[game_id], 0.01, 0.99))
        is_home_side = str(pred.side) == str(game_row.get("home_team", ""))
        model_probability = home_prob if is_home_side else 1.0 - home_prob
        pred.model_probability = float(np.clip(model_probability, 0.01, 0.99))
        pred.edge = pred.model_probability - float(pred.market_implied_probability)
        odds = int(game_row["home_odds"] if is_home_side else game_row["away_odds"])
        pred.expected_value = expected_value(pred.model_probability, odds)
        if not isinstance(pred.metadata, dict):
            pred.metadata = {}
        pred.metadata["predicted_home_win_prob"] = home_prob
    return preds


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
        market_probability = american_to_prob(american_odds)
        payout = get_payout(american_odds)
        edge = model_probability - market_probability
        expected_value = (model_probability * payout) - (1 - model_probability)
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
                "away_team": away_team,
                "home_team": home_team,
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

    total_games_processed = 0

    for sport_name in sports_to_run:
        logger.info("Running %s pipeline", sport_name)
        model = choose_model(sport_name)
        model.runtime_model = None
        historical, daily = load_historical_and_daily(sport_name)
        daily = _ensure_runtime_prediction_columns(daily)
        print("[DEBUG] Daily columns BEFORE prediction:", list(daily.columns))
        runtime_home_win_model = None
        total_games_processed += len(daily)
        csv_path = historical_file_path(sport_name)
        print(f"[{sport_name.upper()}] Looking for historical CSV at: {csv_path}")
        print(f"[{sport_name.upper()}] Exists: {csv_path.exists()}")

        artifact_path = model_artifact_path(sport_name)
        trained_in_run = False
        if csv_path.exists():
            try:
                historical_df = pd.read_csv(csv_path)
                if "home_moneyline" not in historical_df.columns and "closing_moneyline_home" in historical_df.columns:
                    historical_df["home_moneyline"] = historical_df["closing_moneyline_home"]
                if "spread" not in historical_df.columns:
                    if "spread_line" in historical_df.columns:
                        historical_df["spread"] = historical_df["spread_line"]
                    elif "closing_spread_home" in historical_df.columns:
                        historical_df["spread"] = historical_df["closing_spread_home"]
                print(f"[{sport_name.upper()}] Historical rows loaded: {len(historical_df)}")
                if len(historical_df) < 20:
                    print(f"[{sport_name.upper()}] WARNING: Very small dataset")
                try:
                    runtime_home_win_model = train_runtime_model(historical_df)
                    model.runtime_model = runtime_home_win_model
                    trained_in_run = runtime_home_win_model is not None
                except Exception:
                    runtime_home_win_model = None
                    model.runtime_model = None
                    trained_in_run = False
                logger.info("[%s] Runtime model training completed from historical CSV.", sport_name.upper())
            except Exception:
                logger.exception("[%s] Runtime training failed.", sport_name.upper())
                print(f"[{sport_name.upper()}] Training failed → using fallback model")
        else:
            print(f"[{sport_name.upper()}] Historical CSV missing at {csv_path}")

        if not trained_in_run:
            if artifact_path.exists():
                model.load_artifact(artifact_path)
                logger.warning(
                    "[%s] Falling back to model artifact because runtime training failed or historical CSV is missing: %s",
                    sport_name.upper(),
                    artifact_path,
                )
            else:
                raise RuntimeError(
                    f"[{sport_name.upper()}] Runtime training unavailable and no model artifact present at {artifact_path}."
                )

        if hasattr(model, "runtime_model") and model.runtime_model is not None:
            print(f"[{sport_name.upper()}] Using runtime model for predictions")
            if "home_moneyline" not in daily.columns:
                if "home_odds" in daily.columns:
                    daily["home_moneyline"] = daily["home_odds"]

            if "spread" not in daily.columns:
                if "spread_line" in daily.columns:
                    daily["spread"] = daily["spread_line"]
            preds = _build_runtime_moneyline_predictions(daily, model.runtime_model, sport_name)
        else:
            print(f"[{sport_name.upper()}] Using baseline model for predictions")
            try:
                preds = model.predict_daily(daily)
            except Exception as exc:
                logger.warning("[%s] Baseline model prediction failed: %s", sport_name.upper(), exc)
                preds = []
        preds = _apply_runtime_home_win_probabilities(preds, daily, runtime_home_win_model)
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

    pred_path = out_dir / "predictions.csv"

    if not pred_path.exists() or pred_path.stat().st_size == 0:
        print("No predictions generated today — skipping read.")
        return

    pred_path = out_dir / "predictions.csv"

    try:
        if not pred_path.exists():
            print("No predictions file found — skipping.")
            return

        if pred_path.stat().st_size < 10:
            print("Predictions file is empty — skipping.")
            return

        loaded_predictions_df = pd.read_csv(pred_path)

        if loaded_predictions_df.empty or len(loaded_predictions_df.columns) == 0:
            print("Predictions dataframe empty — skipping.")
            return

    except pd.errors.EmptyDataError:
        print("Predictions CSV has no columns — skipping.")
        return
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

    bets_df = pd.DataFrame(bets)
    print("Columns before bet selection:", bets_df.columns.tolist())
    if not bets_df.empty and {"expected_value", "edge"}.issubset(bets_df.columns):
        bets = bets_df[
            (bets_df["expected_value"] > 0.02) &
            (bets_df["edge"] > 0.01)
        ].sort_values(by="expected_value", ascending=False).head(5)
    else:
        bets = bets_df.head(5)

    print("[FINAL BETS]")
    print(
        bets.reindex(
            columns=[
                "away_team",
                "home_team",
                "odds",
                "model_probability",
                "market_probability",
                "edge",
                "expected_value",
            ]
        )
    )

    ranked_bets = bets.to_dict("records")
    final_bets = ranked_bets
    logger.info("Total bets exported: %s", len(final_bets))

    recommendations_df = pd.DataFrame(final_bets)
    recommendations_df = recommendations_df.reindex(columns=RECOMMENDATION_COLUMNS)
    save_dataframe(recommendations_df, out_dir / "recommended_bets.csv")

    if final_bets:
        bt = recommendations_df.copy()
        bt["recommended_units"] = 1.0
        bt["result_units"] = bt["expected_value"] * bt["recommended_units"]
        bt["stake_units"] = bt["recommended_units"]
        bt_summary = summarize_backtest(bt)
        save_dataframe(pd.DataFrame([asdict(bt_summary)]), out_dir / "backtest_summary.csv")

    if len(final_bets) == 0:
        print("No bets exported today")
    else:
        print(f"Exporting {len(final_bets)} bets")
    card = "Top model bets exported from predictions." if final_bets else "No bets exported today"
    report_lines = [
        f"Output directory: {out_dir.resolve()}",
        f"Games processed: {total_games_processed}",
        f"Total ranked recommendations available: {len(ranked_bets)}",
        f"Total recommendations exported (top {TOP_BETS_DAILY}): {len(final_bets)}",
        "",
    ]
    if final_bets:
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
