"""NBA model implementation with boosted ensembles and calibrated probabilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from sports_betting.models.entities import Prediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.odds import (
    american_to_implied_probability,
    expected_value,
    remove_vig_two_way,
)


class EnsembleProbabilityCalibrator:
    """Hybrid Platt + isotonic probability calibrator."""

    def __init__(self, lower: float = 0.08, upper: float = 0.92) -> None:
        self.lower = lower
        self.upper = upper
        self._platt = LogisticRegression(max_iter=500)
        self._isotonic = None
        self._platt_fitted = False
        self._isotonic_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        clipped = np.clip(probs, 1e-4, 1 - 1e-4)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

        # Platt scaling
        self._platt.fit(logits, y_true)
        self._platt_fitted = True

        # Isotonic scaling
        from sklearn.isotonic import IsotonicRegression

        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(clipped, y_true)
        self._isotonic_fitted = True

    def predict(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-4, 1 - 1e-4)
        pred_parts = []

        if self._platt_fitted:
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            pred_parts.append(self._platt.predict_proba(logits)[:, 1])

        if self._isotonic_fitted and self._isotonic is not None:
            pred_parts.append(self._isotonic.predict(clipped))

        if not pred_parts:
            pred_parts.append(clipped)

        blended = np.mean(np.vstack(pred_parts), axis=0)
        return np.clip(blended, self.lower, self.upper)


class NBAModel(SportModel):
    sport = "nba"

    WIN_FEATURES = [
        "elo_diff",
        "offensive_rating_diff",
        "defensive_rating_diff",
        "net_rating_diff",
        "pace_diff",
        "rest_diff",
        "travel_fatigue_diff",
        "recent_form_diff",
        "injury_impact_diff",
        "market_spread",
        "market_total",
    ]

    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "pace_diff",
        "offensive_rating_diff",
        "defensive_rating_diff",
        "net_rating_diff",
        "recent_form_diff",
        "injury_impact_diff",
        "market_spread",
        "market_total",
        "total_line",
    ]

    def __init__(self) -> None:
        self.moneyline_models = self._build_ensemble_models()
        self.spread_models = self._build_ensemble_models()
        self.total_models = self._build_ensemble_models()
        self.moneyline_cal = EnsembleProbabilityCalibrator()
        self.spread_cal = EnsembleProbabilityCalibrator()
        self.total_cal = EnsembleProbabilityCalibrator()
        self.metrics: dict[str, float] = {}

    def _build_ensemble_models(self) -> list:
        models = [RandomForestClassifier(n_estimators=400, min_samples_leaf=8, random_state=42)]
        try:
            from xgboost import XGBClassifier

            models.append(
                XGBClassifier(
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                )
            )
        except Exception:
            try:
                from lightgbm import LGBMClassifier

                models.append(
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.04,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        verbosity=-1,
                    )
                )
            except Exception:
                pass
        return models

    def _build_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{feature: row.get(feature, 0.0) for feature in features} for _, row in df.iterrows()]).fillna(0.0)

    def _ensemble_predict(self, models: list, x: pd.DataFrame) -> np.ndarray:
        probs = [model.predict_proba(x)[:, 1] for model in models]
        return np.mean(np.vstack(probs), axis=0)

    def _fit_market(self, df: pd.DataFrame, features: list[str], target_col: str, models: list, calibrator: EnsembleProbabilityCalibrator) -> tuple[float, float]:
        x = self._build_features(df, features)
        y = df[target_col].astype(int)
        x_train, x_cal, y_train, y_cal = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

        for model in models:
            model.fit(x_train, y_train)

        raw_cal_probs = self._ensemble_predict(models, x_cal)
        calibrator.fit(raw_cal_probs, y_cal.values)

        raw_probs = self._ensemble_predict(models, x)
        probs = calibrator.predict(raw_probs)
        return brier_score_loss(y, probs), log_loss(y, probs)

    def train(self, historical_df: pd.DataFrame) -> None:
        if len(historical_df) < 150:
            raise ValueError("Not enough historical samples to train robustly.")

        ml_brier, ml_log = self._fit_market(historical_df, self.WIN_FEATURES, "home_win", self.moneyline_models, self.moneyline_cal)
        sp_brier, sp_log = self._fit_market(historical_df, self.SPREAD_FEATURES, "home_cover", self.spread_models, self.spread_cal)
        tt_brier, tt_log = self._fit_market(historical_df, self.TOTAL_FEATURES, "over_hit", self.total_models, self.total_cal)

        self.metrics = {
            "moneyline_brier": ml_brier,
            "moneyline_logloss": ml_log,
            "spread_brier": sp_brier,
            "spread_logloss": sp_log,
            "total_brier": tt_brier,
            "total_logloss": tt_log,
        }

    def save_artifact(self, path: Path) -> None:
        payload = {
            "sport": self.sport,
            "win_features": self.WIN_FEATURES,
            "spread_features": self.SPREAD_FEATURES,
            "total_features": self.TOTAL_FEATURES,
            "moneyline_models": self.moneyline_models,
            "spread_models": self.spread_models,
            "total_models": self.total_models,
            "moneyline_cal": self.moneyline_cal,
            "spread_cal": self.spread_cal,
            "total_cal": self.total_cal,
            "metrics": self.metrics,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load_artifact(self, path: Path) -> None:
        with path.open("rb") as f:
            payload = pickle.load(f)

        if payload.get("sport") != self.sport:
            raise ValueError(f"Model artifact sport mismatch for {path}: expected {self.sport}, got {payload.get('sport')}")

        self.moneyline_models = payload["moneyline_models"]
        self.spread_models = payload["spread_models"]
        self.total_models = payload["total_models"]
        self.moneyline_cal = payload["moneyline_cal"]
        self.spread_cal = payload["spread_cal"]
        self.total_cal = payload["total_cal"]
        self.metrics = payload.get("metrics", {})

    def _confidence(self, edge: float, quality: float, feature_completeness: float) -> float:
        conf = 0.54 + min(0.22, abs(edge) * 2.5) + 0.14 * quality + 0.10 * feature_completeness
        return float(np.clip(conf, 0.05, 0.95))

    def _safe_probability(self, value: float) -> float:
        return float(np.clip(value, 0.08, 0.92))

    def _feature_vector_from_row(self, row: pd.Series, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{feature: row.get(feature, 0.0) for feature in features}]).fillna(0.0)

    def predict_daily(self, daily_df: pd.DataFrame) -> list[Prediction]:
        preds: list[Prediction] = []
        for _, row in daily_df.iterrows():
            game_id = row["game_id"]
            away_team, home_team = row["away_team"], row["home_team"]
            game_txt = f"{away_team} @ {home_team}"

            win_features = self._feature_vector_from_row(row, self.WIN_FEATURES)
            spread_features = self._feature_vector_from_row(row, self.SPREAD_FEATURES)
            total_features = self._feature_vector_from_row(row, self.TOTAL_FEATURES)

            completeness = float(1 - np.mean(pd.isna([row.get(feature) for feature in self.WIN_FEATURES])))

            ml_raw = self._ensemble_predict(self.moneyline_models, win_features)
            p_home_ml = self._safe_probability(float(self.moneyline_cal.predict(ml_raw)[0]))
            p_away_ml = self._safe_probability(1 - p_home_ml)
            market_home = american_to_implied_probability(int(row["home_odds"]))
            market_away = american_to_implied_probability(int(row["away_odds"]))
            novig_away, novig_home = remove_vig_two_way(market_away, market_home)

            for side, p_model, p_market, odds in [
                (home_team, p_home_ml, novig_home, int(row["home_odds"])),
                (away_team, p_away_ml, novig_away, int(row["away_odds"])),
            ]:
                edge = p_model - p_market
                ev = expected_value(p_model, odds)
                conf = self._confidence(edge, 1 - self.metrics.get("moneyline_brier", 0.25), completeness)
                reason = (
                    f"Elo diff {row.get('elo_diff', 0.0):.2f}, rest diff {row.get('rest_diff', 0.0):.1f}, "
                    f"injury impact {row.get('injury_impact_diff', 0.0):.2f}, market-model gap {edge:.1%}."
                )
                preds.append(Prediction(game_id, self.sport, "moneyline", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

            sp_raw = self._ensemble_predict(self.spread_models, spread_features)
            p_home_cover = self._safe_probability(float(self.spread_cal.predict(sp_raw)[0]))
            p_away_cover = self._safe_probability(1 - p_home_cover)
            mk_away = american_to_implied_probability(int(row["away_spread_odds"]))
            mk_home = american_to_implied_probability(int(row["home_spread_odds"]))
            nv_away, nv_home = remove_vig_two_way(mk_away, mk_home)

            for side, p_model, p_market, odds in [
                (f"{home_team} {row.get('spread_line', 0.0):+.1f}", p_home_cover, nv_home, int(row["home_spread_odds"])),
                (f"{away_team} {(-row.get('spread_line', 0.0)):+.1f}", p_away_cover, nv_away, int(row["away_spread_odds"])),
            ]:
                edge = p_model - p_market
                ev = expected_value(p_model, odds)
                conf = self._confidence(edge, 1 - self.metrics.get("spread_brier", 0.25), completeness)
                reason = f"Spread signal from rating/rest/travel profile, line {row.get('spread_line', 0.0):+.1f}, edge {edge:.1%}."
                preds.append(Prediction(game_id, self.sport, "spread", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

            tt_raw = self._ensemble_predict(self.total_models, total_features)
            p_over = self._safe_probability(float(self.total_cal.predict(tt_raw)[0]))
            p_under = self._safe_probability(1 - p_over)
            mk_over = american_to_implied_probability(int(row["over_odds"]))
            mk_under = american_to_implied_probability(int(row["under_odds"]))
            nv_over, nv_under = remove_vig_two_way(mk_over, mk_under)

            for side, p_model, p_market, odds in [
                (f"Over {row.get('total_line', 0.0):.1f}", p_over, nv_over, int(row["over_odds"])),
                (f"Under {row.get('total_line', 0.0):.1f}", p_under, nv_under, int(row["under_odds"])),
            ]:
                edge = p_model - p_market
                ev = expected_value(p_model, odds)
                conf = self._confidence(edge, 1 - self.metrics.get("total_brier", 0.25), completeness)
                reason = f"Total model uses pace/efficiency/injury + market baseline {row.get('market_total', row.get('total_line', 0.0)):.1f}; edge {edge:.1%}."
                preds.append(Prediction(game_id, self.sport, "total", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

        return preds
