from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURE_COLUMNS = [
    "implied_home_prob",
    "spread",
    "spread_value_signal",
    "elo_diff",
    "rest_diff",
    "injury_impact_diff",
    "net_rating_diff",
    "last5_net_rating_diff",
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
    df["implied_home_prob"] = df["home_moneyline"].apply(american_to_implied_prob)
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, scaler


def predict(model_bundle, games_df):
    df = prepare_df(games_df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS].copy().fillna(0)

    scaler = None
    model = model_bundle
    if isinstance(model_bundle, tuple) and len(model_bundle) == 2:
        model, scaler = model_bundle

    if scaler is not None:
        X = scaler.transform(X)

    return model.predict_proba(X)[:, 1]
