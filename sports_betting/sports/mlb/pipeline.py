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

DEFAULT_MLB_ERA = 4.20
MLB_REAL_ERA_NORMAL_THRESHOLD = 50.0
MLB_REAL_ERA_SEVERE_THRESHOLD = 25.0


def _is_non_default_era(series: pd.Series) -> pd.Series:
    era = pd.to_numeric(series, errors="coerce")
    return era.notna() & ~np.isclose(era, DEFAULT_MLB_ERA)


def _mlb_quality_from_real_coverage(real_coverage_pct: float) -> str:
    if real_coverage_pct < MLB_REAL_ERA_SEVERE_THRESHOLD:
        return "severe"
    if real_coverage_pct < MLB_REAL_ERA_NORMAL_THRESHOLD:
        return "degraded"
    return "normal"



def _safe_fillna(value: object, default: float) -> object:
    if isinstance(value, pd.Series):
        return value.fillna(default)
    return default if pd.isna(value) else value


def _series_from_get(df: pd.DataFrame, column: str, default: object = 0.0) -> pd.Series:
    series = df.get(column, pd.Series(index=df.index, dtype=object))
    return series if default is None else series.fillna(default)


MLB_TEAM_ALIASES = {
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "athletics": "athletics",
    "oakland a s": "athletics",
    "oakland as": "athletics",
    "st louis cardinals": "st louis cardinals",
}


def normalize_team(team: object) -> str:
    cleaned = str(team or "").lower().strip()
    cleaned = cleaned.replace(".", "").replace("'", "")
    cleaned = " ".join("".join(ch if ch.isalnum() else " " for ch in cleaned).split())
    return MLB_TEAM_ALIASES.get(cleaned, cleaned)


def normalize_pitcher_name(name: object) -> str:
    return " ".join(str(name or "").lower().strip().split())


def _attach_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pitchers_dict = get_probable_pitchers()
    pitchers_dict = {normalize_team(k): v for k, v in pitchers_dict.items()}

    out["home_team_norm"] = out["home_team"].apply(normalize_team)
    out["away_team_norm"] = out["away_team"].apply(normalize_team)

    print("🚨 PITCHERS SCRAPED:", pitchers_dict)
    print("🚨 SAMPLE HOME TEAMS:", out["home_team"].unique()[:5])
    print("🚨 SAMPLE PITCHER KEYS:", list(pitchers_dict.keys())[:5])

    existing_home_pitcher = out.get("pitcher_home", out.get("home_pitcher", out.get("pitcher_name_home", pd.Series("", index=out.index))))
    existing_away_pitcher = out.get("pitcher_away", out.get("away_pitcher", out.get("pitcher_name_away", pd.Series("", index=out.index))))
    out["pitcher_home"] = out["home_team_norm"].map(pitchers_dict)
    out["pitcher_away"] = out["away_team_norm"].map(pitchers_dict)
    out["pitcher_home"] = out["pitcher_home"].where(out["pitcher_home"].notna() & out["pitcher_home"].astype(str).str.strip().ne(""), existing_home_pitcher)
    out["pitcher_away"] = out["pitcher_away"].where(out["pitcher_away"].notna() & out["pitcher_away"].astype(str).str.strip().ne(""), existing_away_pitcher)

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
        clean_era_map[normalize_pitcher_name(pitcher)] = era_value
    era_map = clean_era_map
    home_pitcher_keys = out["pitcher_home"].map(normalize_pitcher_name)
    away_pitcher_keys = out["pitcher_away"].map(normalize_pitcher_name)
    missing_pitchers = set(home_pitcher_keys[home_pitcher_keys.ne("")].dropna()) - set(era_map.keys())
    print("\n[MISSING PITCHERS]:", missing_pitchers)

    existing_home_era = pd.to_numeric(out.get("pitcher_era_home", pd.Series(np.nan, index=out.index)), errors="coerce").replace(0, np.nan)
    existing_away_era = pd.to_numeric(out.get("pitcher_era_away", pd.Series(np.nan, index=out.index)), errors="coerce").replace(0, np.nan)
    mapped_home_era = pd.to_numeric(home_pitcher_keys.map(era_map), errors="coerce")
    mapped_away_era = pd.to_numeric(away_pitcher_keys.map(era_map), errors="coerce")
    real_home_from_map = home_pitcher_keys.ne("") & mapped_home_era.notna() & _is_non_default_era(mapped_home_era)
    real_away_from_map = away_pitcher_keys.ne("") & mapped_away_era.notna() & _is_non_default_era(mapped_away_era)

    out["pitcher_era_home"] = mapped_home_era.fillna(existing_home_era)
    out["pitcher_era_away"] = mapped_away_era.fillna(existing_away_era)

    out["pitcher_era_home"] = out["pitcher_era_home"].replace([0, np.inf, -np.inf], np.nan)
    out["pitcher_era_away"] = out["pitcher_era_away"].replace([0, np.inf, -np.inf], np.nan)

    out["pitcher_era_home_is_real"] = real_home_from_map & out["pitcher_era_home"].notna() & _is_non_default_era(out["pitcher_era_home"])
    out["pitcher_era_away_is_real"] = real_away_from_map & out["pitcher_era_away"].notna() & _is_non_default_era(out["pitcher_era_away"])

    total_games = int(len(out))
    real_home_era_count = int(out["pitcher_era_home_is_real"].sum())
    real_away_era_count = int(out["pitcher_era_away_is_real"].sum())
    real_both_era_count = int((out["pitcher_era_home_is_real"] & out["pitcher_era_away_is_real"]).sum())
    real_pitcher_coverage_pct = (real_both_era_count / total_games * 100.0) if total_games else 0.0
    out["real_home_era_count"] = real_home_era_count
    out["real_away_era_count"] = real_away_era_count
    out["real_both_era_count"] = real_both_era_count
    out["real_pitcher_coverage_pct"] = real_pitcher_coverage_pct
    out["mlb_pitcher_coverage_pct"] = real_pitcher_coverage_pct
    out["pitcher_coverage_pct"] = real_pitcher_coverage_pct
    out["data_quality_status"] = _mlb_quality_from_real_coverage(real_pitcher_coverage_pct)

    out["pitcher_era_home"] = out["pitcher_era_home"].fillna(DEFAULT_MLB_ERA)
    out["pitcher_era_away"] = out["pitcher_era_away"].fillna(DEFAULT_MLB_ERA)

    out["pitcher_era_home"] = out["pitcher_era_home"].clip(lower=1.5, upper=8.0)
    out["pitcher_era_away"] = out["pitcher_era_away"].clip(lower=1.5, upper=8.0)
    default_era_count = int((~out["pitcher_era_home_is_real"]).sum() + (~out["pitcher_era_away_is_real"]).sum())
    out["default_era_count"] = default_era_count
    print("[MLB REAL PITCHER ERA COVERAGE]")
    print("total_games:", total_games)
    print("real both ERAs:", f"{real_both_era_count}/{total_games}")
    print("real home ERAs:", f"{real_home_era_count}/{total_games}")
    print("real away ERAs:", f"{real_away_era_count}/{total_games}")
    print("default ERA count:", default_era_count)
    print("real coverage pct:", f"{real_pitcher_coverage_pct:.1f}%")
    print("data quality:", out["data_quality_status"].iloc[0] if len(out) else "severe")

    out["pitcher_era_diff"] = out["pitcher_era_away"] - out["pitcher_era_home"]
    out["pitcher_diff"] = out["pitcher_era_diff"]
    print("\n[ERA MAP SIZE]:", len(era_map))
    print(out[["pitcher_home", "pitcher_era_home"]].head(10))
    print("\n[ERA COVERAGE CHECK]")
    print("Total pitchers:", len(out))
    print("Non-default ERA count:", (out["pitcher_era_home"] != DEFAULT_MLB_ERA).sum())
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
        frame[col] = frame.get(col, pd.Series(index=frame.index, dtype=float)).fillna(DEFAULT_MLB_ERA)

    frame["pitcher_era_diff"] = frame["pitcher_era_away"] - frame["pitcher_era_home"]
    frame["pitcher_diff"] = frame["pitcher_era_diff"]
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
            "pitcher_era_home": float(row.get("pitcher_era_home", DEFAULT_MLB_ERA)),
            "pitcher_era_away": float(row.get("pitcher_era_away", DEFAULT_MLB_ERA)),
            "pitcher_era_home_is_real": bool(row.get("pitcher_era_home_is_real", False)),
            "pitcher_era_away_is_real": bool(row.get("pitcher_era_away_is_real", False)),
            "real_pitcher_coverage_pct": float(row.get("real_pitcher_coverage_pct", row.get("mlb_pitcher_coverage_pct", 0.0))),
            "real_both_era_count": int(row.get("real_both_era_count", 0)),
            "real_home_era_count": int(row.get("real_home_era_count", 0)),
            "real_away_era_count": int(row.get("real_away_era_count", 0)),
            "default_era_count": int(row.get("default_era_count", 0)),
            "pitcher_diff": float(row.get("pitcher_diff", 0.0)),
            "pitcher_era_diff": float(row.get("pitcher_era_diff", row.get("pitcher_diff", 0.0))),
            "data_quality_status": str(row.get("data_quality_status", "normal")),
            "mlb_pitcher_coverage_pct": float(row.get("mlb_pitcher_coverage_pct", row.get("pitcher_coverage_pct", 0.0))),
            "pitcher_coverage_pct": float(row.get("pitcher_coverage_pct", row.get("mlb_pitcher_coverage_pct", 0.0))),
            "starter_rating_diff": float(row.get("starter_rating_diff", 0.0)),
        }

        selections = [
            (str(row.get("home_team", "")), home_odds, float(row["home_prob"]), float(market_home)),
            (str(row.get("away_team", "")), away_odds, float(row["away_prob"]), float(market_away)),
        ]
        for selection, odds, model_probability, market_probability in selections:
            edge = (model_probability - market_probability) + float(row.get("pitcher_diff", 0.0)) * 0.015
            confidence = float(np.clip(0.5 + edge, 0.01, 0.99))
            if str(row.get("data_quality_status", "normal")).lower() == "degraded":
                confidence *= 0.75
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
                    "confidence": confidence,
                }
            )

    return candidates
