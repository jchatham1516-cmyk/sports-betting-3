"""NBA model implementation with Gradient Boosting + CV calibration."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sports_betting.models.entities import Prediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.odds import (
    american_to_implied_probability,
    expected_value,
    remove_vig_two_way,
)

LOGGER = logging.getLogger(__name__)


class NBAModel(SportModel):
    sport = "nba"

    BASE_FEATURES = [
        "elo_diff",
        "rest_diff",
        "rest_days_home",
        "rest_days_away",
        "back_to_back_home",
        "back_to_back_away",
        "three_in_four_home",
        "three_in_four_away",
        "travel_distance",
        "travel_fatigue_diff",
        "road_trip_length_home",
        "road_trip_length_away",
        "timezone_shift_home",
        "timezone_shift_away",
        "injury_impact",
        "injury_impact_diff",
        "injury_impact_home",
        "injury_impact_away",
        "starter_out_count_home",
        "starter_out_count_away",
        "star_player_out_flag_home",
        "star_player_out_flag_away",
        "starting_goalie_out_flag_home",
        "starting_goalie_out_flag_away",
        "qb_out_flag_home",
        "qb_out_flag_away",
        "offensive_injury_weight_diff",
        "defensive_injury_weight_diff",
        "offensive_rating_diff",
        "defensive_rating_diff",
        "net_rating_diff",
        "pace",
        "home_indicator",
    ]
    WIN_FEATURES = BASE_FEATURES
    SPREAD_FEATURES = BASE_FEATURES + ["spread_line"]
    TOTAL_FEATURES = BASE_FEATURES + ["total_line"]

    def __init__(self) -> None:
        self.moneyline_model: CalibratedClassifierCV | None = None
        self.spread_model: CalibratedClassifierCV | None = None
        self.total_model: CalibratedClassifierCV | None = None
        self.metrics: dict[str, float] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}

    def _base_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=12,
            random_state=42,
        )

    def _build_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return df.reindex(columns=features, fill_value=0.0).fillna(0.0)

    def _fit_market(self, df: pd.DataFrame, features: list[str], target_col: str, market: str) -> tuple[CalibratedClassifierCV, dict[str, float], dict[str, float]]:
        x = self._build_features(df, features)
        y = df[target_col].astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validated raw predictions (before calibration) for diagnostics.
        cv_probs = cross_val_predict(self._base_estimator(), x, y, cv=cv, method="predict_proba")[:, 1]
        cv_probs = np.clip(cv_probs, 0.01, 0.99)
        diagnostics = {
            f"{market}_cv_brier": float(brier_score_loss(y, cv_probs)),
            f"{market}_cv_log_loss": float(log_loss(y, cv_probs)),
        }

        calibrated = CalibratedClassifierCV(self._base_estimator(), method="isotonic", cv=cv)
        calibrated.fit(x, y)

        fitted_estimator = calibrated.calibrated_classifiers_[0].estimator
        importances = {
            name: float(score)
            for name, score in zip(features, getattr(fitted_estimator, "feature_importances_", np.zeros(len(features))), strict=True)
        }
        top = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]
        LOGGER.info("[%s] %s feature importance top-5: %s", self.sport.upper(), market, top)

        calibrated_probs = np.clip(calibrated.predict_proba(x)[:, 1], 0.01, 0.99)
        diagnostics[f"{market}_calibrated_brier"] = float(brier_score_loss(y, calibrated_probs))
        diagnostics[f"{market}_calibrated_log_loss"] = float(log_loss(y, calibrated_probs))
        return calibrated, diagnostics, importances

    def train(self, historical_df: pd.DataFrame) -> None:
        if len(historical_df) < 150:
            raise ValueError("Not enough historical samples to train robustly.")

        self.moneyline_model, ml_metrics, ml_imp = self._fit_market(historical_df, self.WIN_FEATURES, "home_win", "moneyline")
        self.spread_model, sp_metrics, sp_imp = self._fit_market(historical_df, self.SPREAD_FEATURES, "home_cover", "spread")
        self.total_model, tt_metrics, tt_imp = self._fit_market(historical_df, self.TOTAL_FEATURES, "over_hit", "total")

        self.metrics = {**ml_metrics, **sp_metrics, **tt_metrics}
        self.feature_importance = {"moneyline": ml_imp, "spread": sp_imp, "total": tt_imp}

    def save_artifact(self, path: Path) -> None:
        payload = {
            "sport": self.sport,
            "moneyline_model": self.moneyline_model,
            "spread_model": self.spread_model,
            "total_model": self.total_model,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load_artifact(self, path: Path) -> None:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("sport") != self.sport:
            raise ValueError(f"Model artifact sport mismatch for {path}: expected {self.sport}, got {payload.get('sport')}")
        self.moneyline_model = payload["moneyline_model"]
        self.spread_model = payload["spread_model"]
        self.total_model = payload["total_model"]
        self.metrics = payload.get("metrics", {})
        self.feature_importance = payload.get("feature_importance", {})

    def _confidence(self, edge: float, quality: float, feature_completeness: float) -> float:
        conf = 0.55 + min(0.15, abs(edge) * 2.0) + 0.20 * quality + 0.10 * feature_completeness
        return float(np.clip(conf, 0.05, 0.95))

    def _safe_probability(self, value: float) -> float:
        # Shrink toward 50% and keep typical outputs in realistic 40-70% range.
        shrunk = 0.5 + (value - 0.5) * 0.55
        return float(np.clip(shrunk, 0.40, 0.70))

    def _predict_proba(self, model: CalibratedClassifierCV | None, features_df: pd.DataFrame) -> float:
        if model is None:
            raise RuntimeError(f"[{self.sport.upper()}] model artifact not loaded/trained.")
        return float(model.predict_proba(features_df)[:, 1][0])

    def predict_daily(self, daily_df: pd.DataFrame) -> list[Prediction]:
        preds: list[Prediction] = []
        for _, row in daily_df.iterrows():
            game_id = row["game_id"]
            away_team, home_team = row["away_team"], row["home_team"]
            game_txt = f"{away_team} @ {home_team}"

            win_features = self._build_features(pd.DataFrame([row]), self.WIN_FEATURES)
            spread_features = self._build_features(pd.DataFrame([row]), self.SPREAD_FEATURES)
            total_features = self._build_features(pd.DataFrame([row]), self.TOTAL_FEATURES)
            completeness = float(1 - np.mean(pd.isna([row.get(feature) for feature in self.BASE_FEATURES])))

            p_home_ml = self._safe_probability(self._predict_proba(self.moneyline_model, win_features))
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
                conf = self._confidence(edge, 1 - self.metrics.get("moneyline_calibrated_brier", 0.25), completeness)
                reason = (
                    f"Elo {row.get('elo_diff', 0.0):.2f}; injury-adjusted edge {row.get('injury_impact_diff', 0.0):+.2f} "
                    f"(star flags {int(row.get('star_player_out_flag_away', 0))}-{int(row.get('star_player_out_flag_home', 0))}); "
                    f"rest/travel {row.get('rest_diff', 0.0):+.1f}/{row.get('travel_fatigue_diff', 0.0):+.2f}."
                )
                preds.append(Prediction(game_id, self.sport, "moneyline", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

            p_home_cover = self._safe_probability(self._predict_proba(self.spread_model, spread_features))
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
                conf = self._confidence(edge, 1 - self.metrics.get("spread_calibrated_brier", 0.25), completeness)
                spread_reason = (
                    f"Top-rotation efficiency diff {row.get('top_5_rotation_impact_sum_diff', row.get('offensive_line_grade_proxy_diff', row.get('top_line_impact_diff', 0.0))):+.2f}; "
                    f"fatigue split B2B {int(row.get('back_to_back_away', 0))}-{int(row.get('back_to_back_home', 0))}."
                )
                preds.append(Prediction(game_id, self.sport, "spread", side, p_model, p_market, edge, ev, conf, spread_reason, metadata={"game": game_txt}))

            p_over = self._safe_probability(self._predict_proba(self.total_model, total_features))
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
                conf = self._confidence(edge, 1 - self.metrics.get("total_calibrated_brier", 0.25), completeness)
                total_reason = (
                    f"Pace/efficiency {row.get('pace', 0.0):.2f}/{row.get('offensive_rating_diff', 0.0):+.2f}; "
                    f"injury/goaltender-QB flags {int(row.get('starting_goalie_out_flag_away', 0) + row.get('qb_out_flag_away', 0))}-"
                    f"{int(row.get('starting_goalie_out_flag_home', 0) + row.get('qb_out_flag_home', 0))}."
                )
                preds.append(Prediction(game_id, self.sport, "total", side, p_model, p_market, edge, ev, conf, total_reason, metadata={"game": game_txt}))

        return preds
