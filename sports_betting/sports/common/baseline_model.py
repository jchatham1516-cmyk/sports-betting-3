"""Disciplined baseline model focused on stable, explainable betting signals."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sports_betting.models.entities import Prediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.odds import american_to_implied_probability, expected_value, remove_vig_two_way

LOGGER = logging.getLogger(__name__)


class DisciplinedBaselineModel(SportModel):
    """Shared baseline across sports with strict signal support and calibration."""

    sport = "generic"
    BASE_FEATURES = [
        "elo_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "travel_distance",
        "injury_impact_diff",
        "offensive_rating_diff",
        "defensive_rating_diff",
        "net_rating_diff",
        "pace_diff",
    ]
    WIN_FEATURES = BASE_FEATURES
    SPREAD_FEATURES = BASE_FEATURES + ["spread_line"]
    TOTAL_FEATURES = BASE_FEATURES + ["total_line", "pace"]

    PROBABILITY_BOUNDS = {
        "moneyline": (0.12, 0.88),
        "spread": (0.18, 0.82),
        "total": (0.18, 0.82),
    }

    def __init__(self) -> None:
        self.moneyline_model: CalibratedClassifierCV | None = None
        self.spread_model: CalibratedClassifierCV | None = None
        self.total_model: CalibratedClassifierCV | None = None
        self.metrics: dict[str, float] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}

    def _base_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=280,
            learning_rate=0.045,
            max_leaf_nodes=25,
            min_samples_leaf=18,
            random_state=42,
        )

    def _build_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return df.reindex(columns=features, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def _ensure_features(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        for col in features:
            if col not in frame.columns:
                frame[col] = 0.0
        return frame

    def _calibrate(self, base, x: pd.DataFrame, y: pd.Series, method: str, cv) -> CalibratedClassifierCV:
        calibrated = CalibratedClassifierCV(base, method=method, cv=cv)
        calibrated.fit(x, y)
        return calibrated

    def _fit_market(self, df: pd.DataFrame, features: list[str], target_col: str, market: str) -> tuple[CalibratedClassifierCV, dict[str, float]]:
        x = self._build_features(df, features)
        y = df[target_col].astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        raw_probs = cross_val_predict(self._base_estimator(), x, y, cv=cv, method="predict_proba")[:, 1]
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

        # Use isotonic when we have enough data, otherwise sigmoid (Platt-style) for stability.
        method = "isotonic" if len(df) >= 1200 else "sigmoid"
        calibrated = self._calibrate(self._base_estimator(), x, y, method=method, cv=cv)
        calibrated_probs = np.clip(calibrated.predict_proba(x)[:, 1], 0.01, 0.99)
        lo, hi = self.PROBABILITY_BOUNDS[market]
        clamped_probs = np.clip(calibrated_probs, lo, hi)

        diagnostics = {
            f"{market}_raw_brier": float(brier_score_loss(y, raw_probs)),
            f"{market}_raw_log_loss": float(log_loss(y, raw_probs)),
            f"{market}_calibrated_brier": float(brier_score_loss(y, calibrated_probs)),
            f"{market}_calibrated_log_loss": float(log_loss(y, calibrated_probs)),
            f"{market}_clamped_brier": float(brier_score_loss(y, clamped_probs)),
            f"{market}_calibration_method": method,
        }
        LOGGER.info("[%s] %s diagnostics: %s", self.sport.upper(), market, diagnostics)
        return calibrated, diagnostics

    def train(self, historical_df: pd.DataFrame) -> None:
        if len(historical_df) < 300:
            raise ValueError("Not enough historical samples to train disciplined baseline.")
        self.moneyline_model, ml_metrics = self._fit_market(historical_df, self.WIN_FEATURES, "home_win", "moneyline")
        self.spread_model, sp_metrics = self._fit_market(historical_df, self.SPREAD_FEATURES, "home_cover", "spread")
        self.total_model, tt_metrics = self._fit_market(historical_df, self.TOTAL_FEATURES, "over_hit", "total")
        self.metrics = {**ml_metrics, **sp_metrics, **tt_metrics}

    def save_artifact(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "sport": self.sport,
                    "moneyline_model": self.moneyline_model,
                    "spread_model": self.spread_model,
                    "total_model": self.total_model,
                    "metrics": self.metrics,
                    "feature_importance": self.feature_importance,
                },
                f,
            )

    def load_artifact(self, path: Path) -> None:
        with path.open("rb") as f:
            artifact = pickle.load(f)
        if artifact.get("sport") != self.sport:
            raise ValueError(f"Artifact sport mismatch: expected {self.sport}, got {artifact.get('sport')}")
        self.moneyline_model = artifact.get("moneyline_model")
        self.spread_model = artifact.get("spread_model")
        self.total_model = artifact.get("total_model")
        self.metrics = artifact.get("metrics", {})
        self.feature_importance = artifact.get("feature_importance", {})

    def _predict_proba(self, model: CalibratedClassifierCV | None, x: pd.DataFrame) -> float:
        if model is None:
            raise RuntimeError(f"[{self.sport.upper()}] Model not trained/loaded")
        return float(model.predict_proba(self._build_features(x, list(x.columns)))[:, 1][0])

    def _safe_probability(self, probability: float, market: str) -> float:
        lo, hi = self.PROBABILITY_BOUNDS[market]
        return float(np.clip(probability, lo, hi))

    def _support_signals(self, row: pd.Series, p_model: float, p_market: float, market: str, odds: int) -> tuple[int, dict[str, float]]:
        signals = {
            "team_strength_edge": float(abs(float(row.get("elo_diff", 0.0))) >= 35),
            "injury_edge": float(abs(float(row.get("injury_impact_diff", 0.0))) >= 0.8),
            "travel_rest_edge": float(abs(float(row.get("rest_diff", 0.0))) >= 1 or abs(float(row.get("travel_fatigue_diff", 0.0))) >= 0.45),
            "efficiency_mismatch": float(abs(float(row.get("net_rating_diff", 0.0))) >= 1.5),
            "market_disagreement": float(abs(p_model - p_market) >= 0.03),
        }
        if market == "moneyline" and odds >= 180:
            signals["underdog_penalty"] = 1.0
        support_count = int(sum(v for k, v in signals.items() if k != "underdog_penalty"))
        return support_count, signals

    def _confidence(self, edge: float, support_count: int, market: str) -> float:
        brier = self.metrics.get(f"{market}_clamped_brier", self.metrics.get(f"{market}_calibrated_brier", 0.25))
        reliability = max(0.0, 1.0 - brier)
        return float(np.clip(0.35 + 0.9 * max(edge, 0.0) + 0.08 * support_count + 0.35 * reliability, 0.01, 0.99))

    def _reason(self, row: pd.Series, support: dict[str, float], market: str) -> str:
        return (
            f"strength {row.get('elo_diff', 0.0):+.1f}; injury {row.get('injury_impact_diff', 0.0):+.2f}; "
            f"rest/travel {row.get('rest_diff', 0.0):+.1f}/{row.get('travel_fatigue_diff', 0.0):+.2f}; "
            f"efficiency {row.get('net_rating_diff', 0.0):+.2f}; market disagreement {support.get('market_disagreement', 0.0):.0f}."
        )

    def _prediction_metadata(self, row: pd.Series, support_count: int, signals: dict[str, float]) -> dict[str, float | int]:
        return {
            "support_count": support_count,
            **signals,
            "strength_edge": float(row.get("elo_diff", 0.0)),
            "injury_edge": float(row.get("injury_impact_diff", 0.0)),
            "rest_edge": float(row.get("rest_diff", 0.0)),
            "travel_edge": float(row.get("travel_fatigue_diff", 0.0)),
            "efficiency_edge": float(row.get("net_rating_diff", 0.0)),
        }

    def predict_daily(self, daily_df: pd.DataFrame) -> list[Prediction]:
        preds: list[Prediction] = []
        for _, row in daily_df.iterrows():
            game_id = str(row.get("game_id"))
            home_team = str(row.get("home_team"))
            away_team = str(row.get("away_team"))
            game_txt = f"{away_team} @ {home_team}"

            game_frame = pd.DataFrame([row])
            game_frame = self._ensure_features(game_frame, self.WIN_FEATURES)
            p_home_ml = self._safe_probability(self._predict_proba(self.moneyline_model, game_frame[self.WIN_FEATURES]), "moneyline")
            p_away_ml = self._safe_probability(1 - p_home_ml, "moneyline")
            mk_home = american_to_implied_probability(int(row["home_odds"]))
            mk_away = american_to_implied_probability(int(row["away_odds"]))
            nv_home, nv_away = remove_vig_two_way(mk_home, mk_away)

            moneyline_sides = [
                (home_team, p_home_ml, nv_home, int(row["home_odds"])),
                (away_team, p_away_ml, nv_away, int(row["away_odds"])),
            ]
            for side, p_model, p_market, odds in moneyline_sides:
                edge = p_model - p_market
                support_count, signals = self._support_signals(row, p_model, p_market, "moneyline", odds)
                flags = [] if support_count >= 2 else ["low_feature_support"]
                if odds >= 180 and support_count < 4:
                    flags.append("extreme_underdog_guardrail")
                preds.append(
                    Prediction(
                        game_id,
                        self.sport,
                        "moneyline",
                        side,
                        p_model,
                        p_market,
                        edge,
                        expected_value(p_model, odds),
                        self._confidence(edge, support_count, "moneyline"),
                        self._reason(row, signals, "moneyline"),
                        flags,
                        p_market,
                        {"game": game_txt, **self._prediction_metadata(row, support_count, signals)},
                    )
                )

            game_frame = self._ensure_features(game_frame, self.SPREAD_FEATURES)
            p_home_cover = self._safe_probability(self._predict_proba(self.spread_model, game_frame[self.SPREAD_FEATURES]), "spread")
            p_away_cover = self._safe_probability(1 - p_home_cover, "spread")
            mk_hs = american_to_implied_probability(int(row["home_spread_odds"]))
            mk_as = american_to_implied_probability(int(row["away_spread_odds"]))
            nv_away_s, nv_home_s = remove_vig_two_way(mk_as, mk_hs)

            spread_sides = [
                (f"{home_team} {row.get('spread_line', 0.0):+.1f}", p_home_cover, nv_home_s, int(row["home_spread_odds"])),
                (f"{away_team} {(-row.get('spread_line', 0.0)):+.1f}", p_away_cover, nv_away_s, int(row["away_spread_odds"])),
            ]
            for side, p_model, p_market, odds in spread_sides:
                edge = p_model - p_market
                support_count, signals = self._support_signals(row, p_model, p_market, "spread", odds)
                flags = [] if support_count >= 2 else ["low_feature_support"]
                preds.append(
                    Prediction(
                        game_id,
                        self.sport,
                        "spread",
                        side,
                        p_model,
                        p_market,
                        edge,
                        expected_value(p_model, odds),
                        self._confidence(edge, support_count, "spread"),
                        self._reason(row, signals, "spread"),
                        flags,
                        p_market,
                        {"game": game_txt, **self._prediction_metadata(row, support_count, signals)},
                    )
                )

            game_frame = self._ensure_features(game_frame, self.TOTAL_FEATURES)
            p_over = self._safe_probability(self._predict_proba(self.total_model, game_frame[self.TOTAL_FEATURES]), "total")
            p_under = self._safe_probability(1 - p_over, "total")
            mk_over = american_to_implied_probability(int(row["over_odds"]))
            mk_under = american_to_implied_probability(int(row["under_odds"]))
            nv_over, nv_under = remove_vig_two_way(mk_over, mk_under)

            for side, p_model, p_market, odds in [
                (f"Over {row.get('total_line', 0.0):.1f}", p_over, nv_over, int(row["over_odds"])),
                (f"Under {row.get('total_line', 0.0):.1f}", p_under, nv_under, int(row["under_odds"])),
            ]:
                edge = p_model - p_market
                support_count, signals = self._support_signals(row, p_model, p_market, "total", odds)
                flags = [] if support_count >= 2 else ["low_feature_support"]
                preds.append(
                    Prediction(
                        game_id,
                        self.sport,
                        "total",
                        side,
                        p_model,
                        p_market,
                        edge,
                        expected_value(p_model, odds),
                        self._confidence(edge, support_count, "total"),
                        self._reason(row, signals, "total"),
                        flags,
                        p_market,
                        {"game": game_txt, **self._prediction_metadata(row, support_count, signals)},
                    )
                )

        return preds
