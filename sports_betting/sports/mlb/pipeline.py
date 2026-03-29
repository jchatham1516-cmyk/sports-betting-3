"""MLB candidate generation pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_betting.data_collection.espn_mlb_pitchers import fetch_mlb_probable_pitchers_espn
from sports_betting.sports.common.odds import american_to_implied_probability, expected_value, remove_vig_two_way

from .features import build_mlb_features, enrich_mlb_live_features
from .model import MLBModelBundle, predict_mlb_model, save_mlb_model_bundle, train_mlb_model


LOGGER = logging.getLogger(__name__)


def _normalize_team_name(value: object) -> str:
    cleaned = ''.join(ch.lower() if (str(ch).isalnum() or ch.isspace()) else ' ' for ch in str(value or '').strip())
    return ' '.join(cleaned.split())


def _attach_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_team_norm" not in out.columns:
        out["home_team_norm"] = out["home_team"].map(_normalize_team_name)
    if "away_team_norm" not in out.columns:
        out["away_team_norm"] = out["away_team"].map(_normalize_team_name)

    pitcher_df = fetch_mlb_probable_pitchers_espn()

    print("\n[MLB] Pitcher DF Loaded:")
    print(pitcher_df.head())
    print("Rows:", len(pitcher_df))

    if pitcher_df is None or pitcher_df.empty:
        print("WARNING: Pitcher data is EMPTY")

    try:
        merged = out.merge(
            pitcher_df,
            on=["home_team_norm", "away_team_norm"],
            how="left",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[MLB MERGE WARNING] Pitcher merge failed: {exc}")
        merged = out.copy()
        merged["home_pitcher"] = None
        merged["away_pitcher"] = None

    if "home_pitcher" not in merged.columns:
        merged["home_pitcher"] = None
    if "away_pitcher" not in merged.columns:
        merged["away_pitcher"] = None

    print("\n[MLB MERGE CHECK]")
    merge_cols = [col for col in ["home_team", "away_team", "home_pitcher", "away_pitcher"] if col in merged.columns]
    print(merged[merge_cols].head(10))

    merged["pitcher_era_home"] = merged["home_pitcher"].apply(lambda x: 4.2 if pd.isna(x) else 3.5)
    merged["pitcher_era_away"] = merged["away_pitcher"].apply(lambda x: 4.2 if pd.isna(x) else 3.5)
    merged["pitcher_diff"] = merged["pitcher_era_home"] - merged["pitcher_era_away"]
    return merged


def _ensure_daily_mlb_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = out.get("game_id", out.get("event_id", "")).astype(str)
    out["home_team"] = out.get("home_team", "").astype(str)
    out["away_team"] = out.get("away_team", "").astype(str)
    out["event_date"] = out.get("event_date", out.get("commence_time"))
    out["commence_time"] = out.get("commence_time", out.get("event_date"))
    out["home_odds"] = pd.to_numeric(out.get("home_odds", out.get("home_moneyline", 0)), errors="coerce").fillna(0).astype(int)
    out["away_odds"] = pd.to_numeric(out.get("away_odds", 0), errors="coerce").fillna(0).astype(int)
    out["home_moneyline"] = pd.to_numeric(out.get("home_moneyline", out["home_odds"]), errors="coerce").fillna(0.0)
    out["spread"] = pd.to_numeric(out.get("spread", out.get("spread_line", 0.0)), errors="coerce").fillna(0.0)
    out["market_prob"] = pd.to_numeric(
        out.get("market_prob", out["home_odds"].apply(american_to_implied_probability)),
        errors="coerce",
    ).fillna(0.5)
    out["implied_home_prob"] = pd.to_numeric(
        out.get("implied_home_prob", out["home_moneyline"].apply(american_to_implied_probability)),
        errors="coerce",
    ).fillna(0.5)
    out["injury_impact_home"] = pd.to_numeric(out.get("injury_impact_home", 0.0), errors="coerce").fillna(0.0)
    out["injury_impact_away"] = pd.to_numeric(out.get("injury_impact_away", 0.0), errors="coerce").fillna(0.0)
    out = enrich_mlb_live_features(out)
    return build_mlb_features(out)


def run_mlb_pipeline(
    historical_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    model_bundle: MLBModelBundle | None = None,
    artifact_path=None,
) -> list[dict]:
    if daily_df.empty:
        return []

    if model_bundle is None:
        model_bundle = train_mlb_model(historical_df)
        if artifact_path is not None:
            save_mlb_model_bundle(model_bundle, artifact_path)
        LOGGER.info("[MLB] Runtime model training completed from historical CSV.")

    frame = _ensure_daily_mlb_columns(daily_df)
    frame = _attach_pitcher_data(frame)
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
        }

        selections = [
            (str(row.get("home_team", "")), home_odds, float(row["home_prob"]), float(market_home)),
            (str(row.get("away_team", "")), away_odds, float(row["away_prob"]), float(market_away)),
        ]
        for selection, odds, model_probability, market_probability in selections:
            edge = model_probability - market_probability
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
