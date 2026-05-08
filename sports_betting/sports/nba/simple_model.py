from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "implied_home_prob",
    "spread",
    "spread_abs",
    "is_favorite",
    "elo_diff",
    "injury_impact_diff",
    "point_diff_diff",
    "recent_form_diff",
    "recent_form_last5_diff",
    "recent_form_last10_diff",
    "momentum_diff",
    "power_rating_diff",
    "net_rating_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "rest_diff",
    "back_to_back_home",
    "back_to_back_away",
    "back_to_back_diff",
    "three_in_four_home",
    "three_in_four_away",
    "three_in_four_diff",
    "travel_fatigue_diff",
    "home_away_net_rating_split_diff",
    "spread_value_signal",
    "market_implied_probability",
    "line_movement",
]

REQUIRED_INJURY_COLUMNS = [
    "injury_impact_home",
    "injury_impact_away",
    "injury_impact_diff",
]

NEUTRAL_DEFAULTS = {
    "implied_home_prob": 0.5,
    "market_implied_probability": 0.5,
    "home_moneyline": 0.0,
    "spread": 0.0,
    "spread_abs": 0.0,
    "is_favorite": 0.0,
    "elo_home": 1500.0,
    "elo_away": 1500.0,
    "elo_diff": 0.0,
    "net_rating_home": 0.0,
    "net_rating_away": 0.0,
    "net_rating_diff": 0.0,
    "offensive_rating_home": 110.0,
    "offensive_rating_away": 110.0,
    "defensive_rating_home": 110.0,
    "defensive_rating_away": 110.0,
    "offensive_rating_diff": 0.0,
    "defensive_rating_diff": 0.0,
    "point_diff_home": 0.0,
    "point_diff_away": 0.0,
    "point_diff_diff": 0.0,
    "last5_net_rating_home": 0.0,
    "last5_net_rating_away": 0.0,
    "last10_net_rating_home": 0.0,
    "last10_net_rating_away": 0.0,
    "recent_form_diff": 0.0,
    "recent_form_last5_diff": 0.0,
    "recent_form_last10_diff": 0.0,
    "momentum_diff": 0.0,
    "rest_days_home": 3.0,
    "rest_days_away": 3.0,
    "rest_diff": 0.0,
    "back_to_back_home": 0.0,
    "back_to_back_away": 0.0,
    "back_to_back_diff": 0.0,
    "three_in_four_home": 0.0,
    "three_in_four_away": 0.0,
    "three_in_four_diff": 0.0,
    "travel_fatigue_home": 0.0,
    "travel_fatigue_away": 0.0,
    "travel_distance_home": 0.0,
    "travel_distance_away": 0.0,
    "timezone_shift_home": 0.0,
    "timezone_shift_away": 0.0,
    "road_trip_length_home": 0.0,
    "road_trip_length_away": 0.0,
    "travel_fatigue_diff": 0.0,
    "home_away_net_rating_split_diff": 0.0,
    "spread_value_signal": 0.0,
    "line_movement": 0.0,
    "injury_impact_home": 0.0,
    "injury_impact_away": 0.0,
    "injury_impact_diff": 0.0,
}


def _append_missing_columns(df: pd.DataFrame, required_columns: list[str], default: float = 0.0) -> pd.DataFrame:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        filler_df = pd.DataFrame(default, index=df.index, columns=missing_cols)
        df = pd.concat([df, filler_df], axis=1)
    return df


def american_to_implied_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    if pd.isna(odds) or odds == 0:
        return 0.5
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def _num(df: pd.DataFrame, col: str, default: float | pd.Series = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index, dtype=float)


def _coalesce(df: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col in df.columns:
            result = result.fillna(pd.to_numeric(df[col], errors="coerce"))
    return result.fillna(default)


def prepare_df(df):
    df = df.copy()
    df = _append_missing_columns(df, REQUIRED_INJURY_COLUMNS, default=0.0)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"[NBA FEATURE WARNING] Missing features filled with neutral values: {missing}")

    if "home_moneyline" not in df.columns:
        if "home_odds" in df.columns:
            df["home_moneyline"] = df["home_odds"]
        elif "moneyline" in df.columns:
            df["home_moneyline"] = df["moneyline"]
        elif "closing_moneyline_home" in df.columns:
            df["home_moneyline"] = df["closing_moneyline_home"]
        else:
            df["home_moneyline"] = 0
    if "spread" not in df.columns:
        if "spread_line" in df.columns:
            df["spread"] = df["spread_line"]
        elif "closing_spread_home" in df.columns:
            df["spread"] = df["closing_spread_home"]
        else:
            df["spread"] = 0

    for col, default in NEUTRAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="coerce").fillna(0)
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    if "implied_home_prob" not in df.columns or pd.to_numeric(df["implied_home_prob"], errors="coerce").isna().all():
        df["implied_home_prob"] = df["home_moneyline"].apply(american_to_implied_prob)
    else:
        df["implied_home_prob"] = pd.to_numeric(df["implied_home_prob"], errors="coerce").fillna(0.5)
    df["market_implied_probability"] = _coalesce(df, ["market_implied_probability", "market_probability", "market_prob", "implied_home_prob"], 0.5)
    df["spread_abs"] = df["spread"].abs()
    df["is_favorite"] = (df["home_moneyline"] < 0).astype(int)

    df["elo_home"] = _num(df, "elo_home", 1500.0).fillna(1500.0)
    df["elo_away"] = _num(df, "elo_away", 1500.0).fillna(1500.0)
    df["elo_diff"] = _coalesce(df, ["elo_diff"], np.nan).fillna(df["elo_home"] - df["elo_away"])

    df["offensive_rating_home"] = _num(df, "offensive_rating_home", 110.0).replace(0, 110.0).fillna(110.0)
    df["offensive_rating_away"] = _num(df, "offensive_rating_away", 110.0).replace(0, 110.0).fillna(110.0)
    df["defensive_rating_home"] = _num(df, "defensive_rating_home", 110.0).replace(0, 110.0).fillna(110.0)
    df["defensive_rating_away"] = _num(df, "defensive_rating_away", 110.0).replace(0, 110.0).fillna(110.0)
    df["offensive_rating_diff"] = _coalesce(df, ["offensive_rating_diff"], np.nan).fillna(df["offensive_rating_home"] - df["offensive_rating_away"])
    df["defensive_rating_diff"] = _coalesce(df, ["defensive_rating_diff"], np.nan).fillna(df["defensive_rating_away"] - df["defensive_rating_home"])

    df["net_rating_home"] = _num(df, "net_rating_home", 0.0).fillna(0.0)
    df["net_rating_away"] = _num(df, "net_rating_away", 0.0).fillna(0.0)
    df["net_rating_diff"] = _coalesce(df, ["net_rating_diff"], np.nan).fillna(df["net_rating_home"] - df["net_rating_away"])

    df["point_diff_home"] = _num(df, "point_diff_home", 0.0).fillna(0.0)
    df["point_diff_away"] = _num(df, "point_diff_away", 0.0).fillna(0.0)
    df["point_diff_diff"] = df["point_diff_home"] - df["point_diff_away"]

    df["last5_net_rating_home"] = _num(df, "last5_net_rating_home", df["net_rating_home"]).fillna(df["net_rating_home"])
    df["last5_net_rating_away"] = _num(df, "last5_net_rating_away", df["net_rating_away"]).fillna(df["net_rating_away"])
    df["last10_net_rating_home"] = _num(df, "last10_net_rating_home", df["last5_net_rating_home"]).fillna(df["last5_net_rating_home"])
    df["last10_net_rating_away"] = _num(df, "last10_net_rating_away", df["last5_net_rating_away"]).fillna(df["last5_net_rating_away"])
    df["recent_form_last5_diff"] = _coalesce(df, ["recent_form_last5_diff", "last5_net_rating_diff"], np.nan).fillna(df["last5_net_rating_home"] - df["last5_net_rating_away"])
    df["recent_form_last10_diff"] = _coalesce(df, ["recent_form_last10_diff", "last10_net_rating_diff"], np.nan).fillna(df["last10_net_rating_home"] - df["last10_net_rating_away"])
    df["recent_form_diff"] = _coalesce(df, ["recent_form_diff"], np.nan).fillna(df["recent_form_last5_diff"])
    df["momentum_diff"] = (df["recent_form_last5_diff"] - df["recent_form_last10_diff"]).fillna(0.0)

    df["rest_days_home"] = _num(df, "rest_days_home", 3.0).fillna(3.0)
    df["rest_days_away"] = _num(df, "rest_days_away", 3.0).fillna(3.0)
    df["rest_diff"] = _coalesce(df, ["rest_diff"], np.nan).fillna(df["rest_days_home"] - df["rest_days_away"])
    df["back_to_back_home"] = _num(df, "back_to_back_home", 0.0).fillna(0.0)
    df["back_to_back_away"] = _num(df, "back_to_back_away", 0.0).fillna(0.0)
    df["back_to_back_diff"] = df["back_to_back_away"] - df["back_to_back_home"]
    df["three_in_four_home"] = _num(df, "three_in_four_home", 0.0).fillna(0.0)
    df["three_in_four_away"] = _num(df, "three_in_four_away", 0.0).fillna(0.0)
    df["three_in_four_diff"] = df["three_in_four_away"] - df["three_in_four_home"]

    if "travel_fatigue_diff" not in df.columns or pd.to_numeric(df["travel_fatigue_diff"], errors="coerce").isna().all():
        away_fatigue = _num(df, "travel_fatigue_away", 0.0).fillna(0.0) + _num(df, "travel_distance_away", 0.0).fillna(0.0) / 1000.0
        home_fatigue = _num(df, "travel_fatigue_home", 0.0).fillna(0.0) + _num(df, "travel_distance_home", 0.0).fillna(0.0) / 1000.0
        df["travel_fatigue_diff"] = away_fatigue - home_fatigue
    else:
        df["travel_fatigue_diff"] = pd.to_numeric(df["travel_fatigue_diff"], errors="coerce").fillna(0.0)

    df["home_away_net_rating_split_diff"] = _coalesce(df, ["home_away_net_rating_split_diff"], df["net_rating_diff"])
    if "closing_spread_home" in df.columns and "opening_spread_home" in df.columns:
        df["line_movement"] = _num(df, "closing_spread_home") - _num(df, "opening_spread_home")
    else:
        df["line_movement"] = _num(df, "line_movement", 0.0).fillna(0.0)
    df["spread_value_signal"] = _coalesce(df, ["spread_value_signal"], np.nan).fillna(df["spread"] * (df["implied_home_prob"] - 0.5))

    df["injury_impact_diff"] = _coalesce(df, ["injury_impact_diff"], np.nan).fillna(_num(df, "injury_impact_home", 0.0) - _num(df, "injury_impact_away", 0.0))
    df["power_rating_home"] = (df["elo_home"] * 0.6) + (df["net_rating_home"] * 4.0)
    df["power_rating_away"] = (df["elo_away"] * 0.6) + (df["net_rating_away"] * 4.0)
    df["power_rating_diff"] = _coalesce(df, ["power_rating_diff"], np.nan).fillna(df["power_rating_home"] - df["power_rating_away"])

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(NEUTRAL_DEFAULTS.get(col, 0.0))
    for col in df.columns:
        if col not in {"home_team", "away_team", "date", "event_date", "commence_time", "game_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _calibration_bucket_summary(y_true: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    bucket = pd.cut(probs, bins=[0.0, 0.55, 0.60, 0.65, 1.0], labels=["50-55%", "55-60%", "60-65%", "65%+"], include_lowest=True)
    summary = pd.DataFrame({"bucket": bucket, "actual": y_true, "prob": probs}).groupby("bucket", observed=False).agg(
        games=("actual", "size"), avg_probability=("prob", "mean"), win_rate=("actual", "mean")
    ).reset_index()
    return summary


def _evaluate_probabilities(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probs, 0.001, 0.999)
    return {
        "accuracy": float(accuracy_score(y_true, clipped >= 0.5)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
    }


def train_runtime_model(df):
    df = prepare_df(df)
    if "home_win" not in df.columns:
        return None
    if len(df) < 10:
        return None

    X_all = df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_all = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)
    if y_all.nunique() < 2:
        return None

    if "date" in df.columns:
        order = pd.to_datetime(df["date"], errors="coerce").sort_values().index
        split_idx = max(1, int(len(order) * 0.8))
        train_idx, test_idx = order[:split_idx], order[split_idx:]
        split_label = f"chronological 80/20 by date; split date={pd.to_datetime(df.loc[test_idx, 'date'], errors='coerce').min().date() if len(test_idx) else 'n/a'}"
    else:
        split_idx = max(1, int(len(df) * 0.8))
        train_idx, test_idx = df.index[:split_idx], df.index[split_idx:]
        split_label = "row-order 80/20"
    if len(test_idx) == 0 or y_all.loc[test_idx].nunique() < 2:
        train_idx, test_idx = df.index, df.index
        split_label = "in-sample validation fallback (insufficient holdout classes)"

    print(f"[NBA EVAL] training rows: {len(train_idx)}")
    print(f"[NBA EVAL] test rows: {len(test_idx)}")
    print(f"[NBA EVAL] train/test split: {split_label}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all.loc[train_idx])
    X_test = scaler.transform(X_all.loc[test_idx])
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.feature_columns = list(FEATURE_COLUMNS)
    model.scaler = scaler
    model.fit(X_train, y_train)

    raw_probs = model.predict_proba(X_test)[:, 1]
    raw_metrics = _evaluate_probabilities(y_test, raw_probs)
    print(f"[NBA EVAL] validation accuracy: {raw_metrics['accuracy']:.3f}")
    print(f"[NBA EVAL] validation log loss: {raw_metrics['log_loss']:.3f}")
    print(f"[NBA EVAL] validation Brier score: {raw_metrics['brier']:.3f}")

    calibrated_probs = raw_probs
    if len(test_idx) >= 8 and y_test.nunique() == 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_test)
        candidate_probs = np.asarray(calibrator.transform(raw_probs), dtype=float)
        cal_metrics = _evaluate_probabilities(y_test, candidate_probs)
        raw_bucket_gap = _calibration_bucket_summary(y_test, raw_probs).assign(gap=lambda d: (d["avg_probability"] - d["win_rate"]).abs())["gap"].mean()
        cal_bucket_gap = _calibration_bucket_summary(y_test, candidate_probs).assign(gap=lambda d: (d["avg_probability"] - d["win_rate"]).abs())["gap"].mean()
        if cal_metrics["brier"] <= raw_metrics["brier"] or cal_bucket_gap <= raw_bucket_gap:
            model.probability_calibrator = calibrator
            calibrated_probs = candidate_probs
            print(f"[NBA CALIBRATION] isotonic kept; Brier {raw_metrics['brier']:.3f} -> {cal_metrics['brier']:.3f}")
        else:
            model.probability_calibrator = None
            print(f"[NBA CALIBRATION] isotonic rejected; Brier {raw_metrics['brier']:.3f} -> {cal_metrics['brier']:.3f}")
    else:
        model.probability_calibrator = None
        print("[NBA CALIBRATION] skipped; holdout too small or one-class")

    print("[NBA CALIBRATION SUMMARY]")
    print(_calibration_bucket_summary(y_test, calibrated_probs).to_string(index=False))

    importances = pd.Series(np.abs(model.coef_[0]), index=FEATURE_COLUMNS).sort_values(ascending=False)
    model.feature_importances_ = importances
    print("[NBA FEATURE IMPORTANCE TOP 15]")
    print(importances.head(15).to_string())
    total_importance = float(importances.sum())
    if total_importance > 0 and float(importances.head(3).sum() / total_importance) > 0.75:
        print("⚠️ NBA MODEL RELIANCE WARNING: top 3 features account for >75% of importance")
    print("🔥 MODEL FIT COMPLETE")
    print("✅ MODEL TRAINED:", type(model))
    return model


def predict(model_bundle, games_df):
    df = prepare_df(games_df)
    df = _append_missing_columns(df, REQUIRED_INJURY_COLUMNS, default=0.0)

    scaler = getattr(model_bundle, "scaler", None)
    model = model_bundle
    if isinstance(model_bundle, tuple) and len(model_bundle) >= 2:
        model, scaler = model_bundle

    feature_columns = list(getattr(model, "feature_columns", FEATURE_COLUMNS))
    X = df.reindex(columns=feature_columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if scaler is not None:
        X_pred = scaler.transform(X)
    else:
        X_pred = X.values
    probs = np.asarray(model.predict_proba(X_pred)[:, 1], dtype=float)
    calibrator = getattr(model, "probability_calibrator", None)
    if calibrator is not None:
        probs = np.asarray(calibrator.transform(probs), dtype=float)
    return np.clip(probs, 0.01, 0.99)
