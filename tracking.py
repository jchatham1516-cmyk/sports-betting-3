import os
from datetime import datetime

import pandas as pd

TRACKING_FILE = "data/tracking/bet_history.csv"


def log_bets(final_bets):
    file_path = os.path.abspath(TRACKING_FILE)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = final_bets.copy()
    df["date"] = datetime.now().strftime("%Y-%m-%d")
    df["result"] = None
    df["profit"] = None

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(file_path, index=False)
    print("\n[TRACKING DEBUG]")
    print("Saved tracking file to:", file_path)
    print("File exists:", os.path.exists(file_path))
    print("Current working directory:", os.getcwd())
    print("Absolute path:", file_path)
