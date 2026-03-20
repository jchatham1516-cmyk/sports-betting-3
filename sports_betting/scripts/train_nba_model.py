import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_nba_model():
    df = pd.read_csv("sports_betting/data/historical/nba_historical.csv")

    if len(df) < 50:
        print("Not enough data to train model")
        return None

    # target
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # features
    df["ml_home_prob"] = df["closing_moneyline_home"].apply(
        lambda ml: abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
    )

    df["spread"] = df["closing_spread_home"]

    X = df[["ml_home_prob", "spread"]]
    y = df["home_win"]

    model = LogisticRegression()
    # Save feature order if DataFrame
    if hasattr(X, "columns"):
        model.feature_columns = list(X.columns)
        X_train = X.values
    else:
        model.feature_columns = None
        X_train = X

    model.fit(X_train, y)

    return model
