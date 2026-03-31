from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURE_COLUMNS = [
    "implied_home_prob",
    "spread",
    "spread_abs",
    "is_favorite",
    "elo_diff",
    "injury_impact_diff",
    "point_diff_diff",
    "recent_form_diff",
    "momentum_diff",
    "power_rating_diff",
]

REQUIRED_INJURY_COLUMNS = [
    "injury_impact_home",
    "injury_impact_away",
    "injury_impact_diff",
]


def _append_missing_columns(df: pd.DataFrame, required_columns: list[str], default: float = 0.0) -> pd.DataFrame:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        filler_df = pd.DataFrame(default, index=df.index, columns=missing_cols)
        df = pd.concat([df, filler_df], axis=1)
    return df


def _drop_constant_features(df: pd.DataFrame, protected_columns: set[str] | None = None) -> pd.DataFrame:
    protected = protected_columns or set()
    if len(df.index) <= 1:
        return df
    drop_cols: list[str] = []
    for col in df.columns:
        if col in protected:
            continue
        if df[col].nunique(dropna=False) <= 1:
            print(f"⚠️ Dropping useless feature: {col}")
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def american_to_implied_prob(odds):
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def prepare_df(df):
    df = df.copy()
    df = _append_missing_columns(df, REQUIRED_INJURY_COLUMNS, default=0.0)
    if "home_moneyline" not in df.columns:
        if "home_odds" in df.columns:
            df["home_moneyline"] = df["home_odds"]
        elif "moneyline" in df.columns:
            df["home_moneyline"] = df["moneyline"]
        else:
            df["home_moneyline"] = 0
    if "spread" not in df.columns:
        if "spread_line" in df.columns:
            df["spread"] = df["spread_line"]
        else:
            df["spread"] = 0
    df["home_moneyline"] = pd.to_numeric(df["home_moneyline"], errors="coerce").fillna(0)
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    if "implied_home_prob" not in df.columns:
        df["implied_home_prob"] = df["home_moneyline"].apply(american_to_implied_prob)
    else:
        df["implied_home_prob"] = pd.to_numeric(df["implied_home_prob"], errors="coerce").fillna(0)
    df["spread_abs"] = df["spread"].abs()
    df["is_favorite"] = (df["home_moneyline"] < 0).astype(int)
    df["spread_value_signal"] = df["spread"] * df["implied_home_prob"]
    if "injury_impact_diff" not in df.columns:
        df["injury_impact_diff"] = pd.to_numeric(df["injury_impact_home"], errors="coerce").fillna(0) - pd.to_numeric(df["injury_impact_away"], errors="coerce").fillna(0)
    else:
        df["injury_impact_diff"] = pd.to_numeric(df["injury_impact_diff"], errors="coerce").fillna(0)

    if "elo_home" not in df.columns:
        df["elo_home"] = 1500.0
    if "elo_away" not in df.columns:
        df["elo_away"] = 1500.0
    if "net_rating_home" not in df.columns:
        df["net_rating_home"] = 0.0
    if "net_rating_away" not in df.columns:
        df["net_rating_away"] = 0.0
    if "point_diff_home" not in df.columns:
        df["point_diff_home"] = 0.0
    if "point_diff_away" not in df.columns:
        df["point_diff_away"] = 0.0
    if "last5_net_rating_home" not in df.columns:
        df["last5_net_rating_home"] = pd.NA
    if "last5_net_rating_away" not in df.columns:
        df["last5_net_rating_away"] = pd.NA
    if "last10_net_rating_home" not in df.columns:
        df["last10_net_rating_home"] = pd.NA
    if "last10_net_rating_away" not in df.columns:
        df["last10_net_rating_away"] = pd.NA

    df["elo_home"] = pd.to_numeric(df["elo_home"], errors="coerce").fillna(1500.0)
    df["elo_away"] = pd.to_numeric(df["elo_away"], errors="coerce").fillna(1500.0)
    df["net_rating_home"] = pd.to_numeric(df["net_rating_home"], errors="coerce").fillna(0.0)
    df["net_rating_away"] = pd.to_numeric(df["net_rating_away"], errors="coerce").fillna(0.0)
    df["point_diff_home"] = pd.to_numeric(df["point_diff_home"], errors="coerce").fillna(0.0)
    df["point_diff_away"] = pd.to_numeric(df["point_diff_away"], errors="coerce").fillna(0.0)
    df["last5_net_rating_home"] = pd.to_numeric(df["last5_net_rating_home"], errors="coerce").fillna(df["net_rating_home"])
    df["last5_net_rating_away"] = pd.to_numeric(df["last5_net_rating_away"], errors="coerce").fillna(df["net_rating_away"])
    df["last10_net_rating_home"] = pd.to_numeric(df["last10_net_rating_home"], errors="coerce").fillna(df["last5_net_rating_home"])
    df["last10_net_rating_away"] = pd.to_numeric(df["last10_net_rating_away"], errors="coerce").fillna(df["last5_net_rating_away"])

    df["point_diff_diff"] = df["point_diff_home"] - df["point_diff_away"]
    df["recent_form_diff"] = df["last5_net_rating_home"] - df["last5_net_rating_away"]
    df["momentum_diff"] = (
        (df["last10_net_rating_home"] - df["last5_net_rating_home"])
        - (df["last10_net_rating_away"] - df["last5_net_rating_away"])
    )
    df["power_rating_home"] = (df["elo_home"] * 0.6) + (df["net_rating_home"] * 0.4)
    df["power_rating_away"] = (df["elo_away"] * 0.6) + (df["net_rating_away"] * 0.4)
    df["power_rating_diff"] = df["power_rating_home"] - df["power_rating_away"]
    if "elo_diff" not in df.columns:
        df["elo_diff"] = pd.NA
    df["elo_diff"] = pd.to_numeric(df["elo_diff"], errors="coerce").fillna(df["elo_home"] - df["elo_away"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def train_runtime_model(df):
    df = prepare_df(df)
    if "injury_impact_diff" not in df.columns:
        df["injury_impact_diff"] = 0

    if len(df) < 10:
        return None

    df = _append_missing_columns(df, FEATURE_COLUMNS, default=0.0)
    df = _drop_constant_features(df, protected_columns={"home_win"})
    feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not feature_columns:
        return None

    X = df[feature_columns].copy().fillna(0)
    y = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if y.nunique() < 2:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.feature_columns = feature_columns
    model.scaler = scaler
    # Save feature order if DataFrame
    if hasattr(X, "columns"):
        model.feature_columns = list(X.columns)
        X_train = X.values
    else:
        X_train = X

    model.fit(X_train, y)
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

    feature_columns = getattr(model, "feature_columns", FEATURE_COLUMNS)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features at prediction time: {missing}")

    X = df[feature_columns].copy().fillna(0)
    if scaler is not None:
        X = scaler.transform(X)

    if hasattr(model, "feature_columns") and model.feature_columns is not None:
        X_pred = X
    elif hasattr(X, "values"):
        X_pred = X.values
    else:
        X_pred = X

    probs = model.predict_proba(X_pred)[:, 1]
    return probs
