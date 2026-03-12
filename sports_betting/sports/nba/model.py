"""NBA model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from sports_betting.models.entities import Prediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.calibration import IsotonicCalibrator
from sports_betting.sports.common.odds import (
    american_to_implied_probability,
    expected_value,
    remove_vig_two_way,
)


class NBAModel(SportModel):
    sport = "nba"

    WIN_FEATURES = [
        "elo_diff",
        "net_rating_diff",
        "off_rating_diff",
        "def_rating_diff",
        "pace_diff",
        "rest_diff",
        "injury_impact_diff",
        "travel_fatigue_diff",
        "recent_form_diff",
    ]

    SPREAD_FEATURES = WIN_FEATURES + ["spread_line"]
    TOTAL_FEATURES = [
        "pace_sum",
        "off_rating_sum",
        "def_rating_sum",
        "recent_total_trend",
        "injury_total_impact",
        "total_line",
    ]

    def __init__(self) -> None:
        self.moneyline_model = LogisticRegression(max_iter=500)
        self.spread_model = LogisticRegression(max_iter=500)
        self.total_model = LogisticRegression(max_iter=500)
        self.moneyline_cal = IsotonicCalibrator()
        self.spread_cal = IsotonicCalibrator()
        self.total_cal = IsotonicCalibrator()
        self.metrics: dict[str, float] = {}

    def _fit_market(
        self,
        df: pd.DataFrame,
        features: list[str],
        target_col: str,
        model: LogisticRegression,
        calibrator: IsotonicCalibrator,
    ) -> tuple[float, float]:
        x = df[features].fillna(0.0)
        y = df[target_col].astype(int)
        model.fit(x, y)
        raw_probs = model.predict_proba(x)[:, 1]
        calibrator.fit(raw_probs, y.values)
        probs = calibrator.predict(raw_probs)
        return brier_score_loss(y, probs), log_loss(y, probs)

    def train(self, historical_df: pd.DataFrame) -> None:
        if len(historical_df) < 100:
            raise ValueError("Not enough historical NBA samples to train robustly.")

        ml_brier, ml_log = self._fit_market(
            historical_df,
            self.WIN_FEATURES,
            "home_win",
            self.moneyline_model,
            self.moneyline_cal,
        )
        sp_brier, sp_log = self._fit_market(
            historical_df,
            self.SPREAD_FEATURES,
            "home_cover",
            self.spread_model,
            self.spread_cal,
        )
        tt_brier, tt_log = self._fit_market(
            historical_df,
            self.TOTAL_FEATURES,
            "over_hit",
            self.total_model,
            self.total_cal,
        )
        self.metrics = {
            "moneyline_brier": ml_brier,
            "moneyline_logloss": ml_log,
            "spread_brier": sp_brier,
            "spread_logloss": sp_log,
            "total_brier": tt_brier,
            "total_logloss": tt_log,
        }

    def _confidence(self, edge: float, quality: float, feature_completeness: float) -> float:
        conf = 0.50 + min(0.30, abs(edge) * 3.0) + 0.10 * quality + 0.10 * feature_completeness
        return float(np.clip(conf, 0.05, 0.99))

    def predict_daily(self, daily_df: pd.DataFrame) -> list[Prediction]:
        preds: list[Prediction] = []
        for _, row in daily_df.iterrows():
            game_id = row["game_id"]
            away_team, home_team = row["away_team"], row["home_team"]
            game_txt = f"{away_team} @ {home_team}"
            completeness = 1 - row[self.WIN_FEATURES].isna().mean()

            ml_raw = self.moneyline_model.predict_proba(pd.DataFrame([row[self.WIN_FEATURES].fillna(0.0)]))[:, 1]
            p_home_ml = float(self.moneyline_cal.predict(ml_raw)[0])
            p_away_ml = 1 - p_home_ml
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
                    f"Net rating diff {row['net_rating_diff']:.2f}, rest diff {row['rest_diff']:.1f}, "
                    f"injury diff {row['injury_impact_diff']:.2f}, market-model gap {edge:.1%}."
                )
                preds.append(
                    Prediction(game_id, self.sport, "moneyline", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt})
                )

            sp_raw = self.spread_model.predict_proba(pd.DataFrame([row[self.SPREAD_FEATURES].fillna(0.0)]))[:, 1]
            p_home_cover = float(self.spread_cal.predict(sp_raw)[0])
            p_away_cover = 1 - p_home_cover
            mk_away = american_to_implied_probability(int(row["away_spread_odds"]))
            mk_home = american_to_implied_probability(int(row["home_spread_odds"]))
            nv_away, nv_home = remove_vig_two_way(mk_away, mk_home)

            for side, p_model, p_market, odds in [
                (f"{home_team} {row['spread_line']:+.1f}", p_home_cover, nv_home, int(row["home_spread_odds"])),
                (f"{away_team} {(-row['spread_line']):+.1f}", p_away_cover, nv_away, int(row["away_spread_odds"])),
            ]:
                edge = p_model - p_market
                ev = expected_value(p_model, odds)
                conf = self._confidence(edge, 1 - self.metrics.get("spread_brier", 0.25), completeness)
                reason = f"Spread model favored by rating + rest blend; line {row['spread_line']:+.1f}; gap {edge:.1%}."
                preds.append(Prediction(game_id, self.sport, "spread", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

            tt_raw = self.total_model.predict_proba(pd.DataFrame([row[self.TOTAL_FEATURES].fillna(0.0)]))[:, 1]
            p_over = float(self.total_cal.predict(tt_raw)[0])
            p_under = 1 - p_over
            mk_over = american_to_implied_probability(int(row["over_odds"]))
            mk_under = american_to_implied_probability(int(row["under_odds"]))
            nv_over, nv_under = remove_vig_two_way(mk_over, mk_under)
            for side, p_model, p_market, odds in [
                (f"Over {row['total_line']:.1f}", p_over, nv_over, int(row["over_odds"])),
                (f"Under {row['total_line']:.1f}", p_under, nv_under, int(row["under_odds"])),
            ]:
                edge = p_model - p_market
                ev = expected_value(p_model, odds)
                conf = self._confidence(edge, 1 - self.metrics.get("total_brier", 0.25), completeness)
                reason = f"Pace-total dynamics and shooting efficiency trend suggest {side}; gap {edge:.1%}."
                preds.append(Prediction(game_id, self.sport, "total", side, p_model, p_market, edge, ev, conf, reason, metadata={"game": game_txt}))

        return preds
