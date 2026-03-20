import pandas as pd


def show_performance():
    df = pd.read_csv("data/tracking/bet_history.csv")
    df = df.dropna(subset=["result"])

    total = len(df)
    wins = (df["result"] == "win").sum()
    win_rate = wins / total if total else 0

    profit = df["profit"].sum()
    avg_ev = df["expected_value"].mean()

    print("Total Bets:", total)
    print("Win Rate:", win_rate)
    print("Profit:", profit)
    print("Avg EV:", avg_ev)
