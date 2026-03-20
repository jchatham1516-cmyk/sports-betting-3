from sklearn.linear_model import LogisticRegression
import pandas as pd

FEATURE_COLUMNS = [
    "implied_home_prob",
    "spread",
    "spread_value_signal"
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

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        return None

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


def predict(model, games_df):
    df = prepare_df(games_df)

    X = df[FEATURE_COLUMNS].copy().fillna(0)

    return model.predict_proba(X)[:, 1]
