from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURE_COLUMNS = [
    "implied_home_prob",
    "spread",
    "spread_abs",
    "is_favorite",
    "elo_diff",
]


def american_to_implied_prob(odds):
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def prepare_df(df):
    df = df.copy()
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
    return df


def train_runtime_model(df):
    df = prepare_df(df)

    if len(df) < 10:
        return None

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS].copy().fillna(0)
    y = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        return None

    feature_columns = X.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.feature_columns = feature_columns
    model.fit(X, y)

    return model, scaler


def predict(model_bundle, games_df):
    df = prepare_df(games_df)

    scaler = None
    model = model_bundle
    if isinstance(model_bundle, tuple) and len(model_bundle) >= 2:
        model, scaler = model_bundle

    feature_columns = getattr(model, "feature_columns", FEATURE_COLUMNS)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features at prediction time: {missing}")

    X = df[feature_columns].copy().fillna(0)
    if scaler is not None:
        X = scaler.transform(X)

    return model.predict_proba(X.values if isinstance(X, pd.DataFrame) else X)[:, 1]
