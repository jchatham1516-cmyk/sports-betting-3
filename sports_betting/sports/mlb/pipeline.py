"""MLB candidate generation pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_betting.data_collection.mlb_pitchers import get_probable_pitchers
from sports_betting.data_collection.pitcher_stats import build_pitcher_era_map
from sports_betting.sports.common.odds import american_to_implied_probability, expected_value, remove_vig_two_way

from .features import build_mlb_features, enrich_mlb_live_features
from .model import MLBModelBundle, predict_mlb_model, save_mlb_model_bundle, train_mlb_model


LOGGER = logging.getLogger(__name__)


def _safe_fillna(value: object, default: float) -> object:
    if isinstance(value, pd.Series):
        return value.fillna(default)
    return default if pd.isna(value) else value


def _series_from_get(df: pd.DataFrame, column: str, default: object = 0.0) -> pd.Series:
    series = df.get(column, pd.Series(index=df.index, dtype=object))
    return series if default is None else series.fillna(default)


def normalize_team(team: object) -> str:
    return str(team).lower().strip()


def _attach_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pitchers_dict = get_probable_pitchers()
    pitchers_dict = {normalize_team(k): v for k, v in pitchers_dict.items()}

    out["home_team_norm"] = out["home_team"].apply(normalize_team)
    out["away_team_norm"] = out["away_team"].apply(normalize_team)

    print("🚨 PITCHERS SCRAPED:", pitchers_dict)
    print("🚨 SAMPLE HOME TEAMS:", out["home_team"].unique()[:5])
    print("🚨 SAMPLE PITCHER KEYS:", list(pitchers_dict.keys())[:5])

    out["pitcher_home"] = out["home_team_norm"].map(pitchers_dict)
    out["pitcher_away"] = out["away_team_norm"].map(pitchers_dict)

    print("[PITCHER MATCH CHECK]")
    print(out[["home_team", "pitcher_home"]].head(10))

    era_map = build_pitcher_era_map(pitchers_dict)
    clean_era_map: dict[str, float] = {}
    for pitcher, era in era_map.items():
        try:
            era_value = float(era)
        except (TypeError, ValueError):
            continue
        if era_value <= 0 or era_value > 15:
            continue
        clean_era_map[pitcher] = era_value
    era_map = clean_era_map
    print("🚨 ERA MAP:", era_map)

    out["pitcher_era_home"] = out["pitcher_home"].map(era_map)
    out["pitcher_era_away"] = out["pitcher_away"].map(era_map)

    out["pitcher_era_home"] = out["pitcher_era_home"].fillna(4.20)
    out["pitcher_era_away"] = out["pitcher_era_away"].fillna(4.20)
    out.loc[out["pitcher_era_home"] <= 0, "pitcher_era_home"] = 4.20
    out.loc[out["pitcher_era_away"] <= 0, "pitcher_era_away"] = 4.20
    print("[ERA VALIDATION CHECK]")
    print(out[["pitcher_home", "pitcher_era_home"]].head(10))
    print("ERA STATS:")
    print(out["pitcher_era_home"].describe())

    out["pitcher_diff"] = out["pitcher_era_away"] - out["pitcher_era_home"]
    print("[PITCHER DIFF SUMMARY]")
    print(out["pitcher_diff"].describe())
    return out


def _ensure_daily_mlb_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = _series_from_get(out, "game_id", "") if "game_id" in out.columns else _series_from_get(out, "event_id", "")
    out["home_team"] = _series_from_get(out, "home_team", "").astype(str)
    out["away_team"] = _series_from_get(out, "away_team", "").astype(str)
    out["event_date"] = _series_from_get(out, "event_date", None) if "event_date" in out.columns else _series_from_get(out, "commence_time", None)
    out["commence_time"] = _series_from_get(out, "commence_time", None) if "commence_time" in out.columns else _series_from_get(out, "event_date", None)

    home_odds_source = out.get("home_odds", out.get("home_moneyline", pd.Series(index=out.index, dtype=float)))
    away_odds_source = out.get("away_odds", pd.Series(index=out.index, dtype=float))
    market_prob_source = out.get("market_prob", pd.to_numeric(home_odds_source, errors="coerce").apply(american_to_implied_probability))

    out["home_odds"] = _safe_fillna(pd.to_numeric(home_odds_source, errors="coerce"), 0).astype(int)
    out["away_odds"] = _safe_fillna(pd.to_numeric(away_odds_source, errors="coerce"), 0).astype(int)
    out["home_moneyline"] = _safe_fillna(pd.to_numeric(out.get("home_moneyline", out["home_odds"]), errors="coerce"), 0.0)
    out["spread"] = _safe_fillna(pd.to_numeric(out.get("spread", out.get("spread_line", pd.Series(index=out.index, dtype=float))), errors="coerce"), 0.0)
    out["market_prob"] = _safe_fillna(pd.to_numeric(market_prob_source, errors="coerce"), 0.5)
    out["implied_home_prob"] = _safe_fillna(
        pd.to_numeric(out.get("implied_home_prob", out["home_moneyline"].apply(american_to_implied_probability)), errors="coerce"),
        0.5,
    )
    out["injury_impact_home"] = _safe_fillna(pd.to_numeric(out.get("injury_impact_home", pd.Series(index=out.index, dtype=float)), errors="coerce"), 0.0)
    out["injury_impact_away"] = _safe_fillna(pd.to_numeric(out.get("injury_impact_away", pd.Series(index=out.index, dtype=float)), errors="coerce"), 0.0)
    out = enrich_mlb_live_features(out)
    return build_mlb_features(out)


def run_mlb_pipeline(
    historical_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    model_bundle: MLBModelBundle | None = None,
    artifact_path=None,
) -> list[dict]:
    print("⚾ INSIDE MLB PIPELINE")
    df = daily_df
    print(f"[MLB DEBUG] rows before filtering: {len(df)}")
    if daily_df.empty:
        print("🚨 MLB EMPTY DATA — NO GAMES FOUND")
        return []

    if model_bundle is None:
        model_bundle = train_mlb_model(historical_df)
        if artifact_path is not None:
            save_mlb_model_bundle(model_bundle, artifact_path)
        LOGGER.info("[MLB] Runtime model training completed from historical CSV.")

    frame = _ensure_daily_mlb_columns(daily_df)
    frame = _attach_pitcher_data(frame)

    for col in [
        "starter_rating_home",
        "starter_rating_away",
        "bullpen_rating_home",
        "bullpen_rating_away",
        "hitting_rating_home",
        "hitting_rating_away",
    ]:
        frame[col] = frame.get(col, pd.Series(index=frame.index, dtype=float)).fillna(0)

    for col in [
        "pitcher_era_home",
        "pitcher_era_away",
    ]:
        frame[col] = frame.get(col, pd.Series(index=frame.index, dtype=float)).fillna(4.20)

    frame["pitcher_diff"] = frame["pitcher_era_away"] - frame["pitcher_era_home"]
    frame["edge"] = _series_from_get(frame, "edge", 0.0) + (frame["pitcher_diff"] * 0.015)
    frame = frame.fillna(0)

    print("[MLB DEBUG] columns:")
    print(
        frame[
            [
                "pitcher_era_home",
                "pitcher_era_away",
                "pitcher_diff",
            ]
        ].head()
    )

    print("[MLB DEBUG] pitcher_diff summary:")
    print(frame["pitcher_diff"].describe())

    frame = predict_mlb_model(model_bundle, frame)
    frame["home_prob"] = frame["predicted_home_win_prob"].clip(0.01, 0.99)
    frame["away_prob"] = frame["predicted_away_win_prob"].clip(0.01, 0.99)

    candidates: list[dict] = []
    for _, row in frame.iterrows():
        home_odds = int(row.get("home_odds", 0))
        away_odds = int(row.get("away_odds", 0))
        if home_odds == 0 or away_odds == 0:
            continue

        home_market_raw = american_to_implied_probability(home_odds)
        away_market_raw = american_to_implied_probability(away_odds)
        market_home, market_away = remove_vig_two_way(home_market_raw, away_market_raw)

        game = f"{row.get('away_team')} @ {row.get('home_team')}"
        common = {
            "sport": "mlb",
            "event_id": str(row.get("game_id", "")),
            "commence_time": row.get("commence_time", row.get("event_date")),
            "away_team": str(row.get("away_team", "")),
            "home_team": str(row.get("home_team", "")),
            "game": game,
            "market": "moneyline",
            "market_type": "moneyline",
            "support_count": 0,
            "composite_score": 0.0,
            "reason_summary": "MLB runtime moneyline prediction",
            "injury_impact_home": float(row.get("injury_impact_home", 0.0)),
            "injury_impact_away": float(row.get("injury_impact_away", 0.0)),
            "injury_impact_diff": float(row.get("injury_impact_diff", 0.0)),
            "pitcher_home": str(row.get("pitcher_home", "")),
            "pitcher_away": str(row.get("pitcher_away", "")),
            "pitcher_era_home": float(row.get("pitcher_era_home", 4.2)),
            "pitcher_era_away": float(row.get("pitcher_era_away", 4.2)),
            "pitcher_diff": float(row.get("pitcher_diff", 0.0)),
        }

        selections = [
            (str(row.get("home_team", "")), home_odds, float(row["home_prob"]), float(market_home)),
            (str(row.get("away_team", "")), away_odds, float(row["away_prob"]), float(market_away)),
        ]
        for selection, odds, model_probability, market_probability in selections:
            edge = (model_probability - market_probability) + float(row.get("pitcher_diff", 0.0)) * 0.015
            candidates.append(
                {
                    **common,
                    "selection": selection,
                    "odds": int(odds),
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "model_prob": model_probability if selection == str(row.get("home_team", "")) else 1.0 - model_probability,
                    "model_probability": model_probability,
                    "market_probability": market_probability,
                    "edge": edge,
                    "expected_value": expected_value(model_probability, int(odds)),
                    "confidence": float(np.clip(0.5 + edge, 0.01, 0.99)),
                }
            )

    return candidates
