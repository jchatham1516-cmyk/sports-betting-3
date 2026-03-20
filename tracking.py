import os
from datetime import datetime

import pandas as pd

TRACKING_FILE = "data/tracking/bet_history.csv"


def log_bets(final_bets):
    os.makedirs("data/tracking", exist_ok=True)

    df = final_bets.copy()
    df["date"] = datetime.now().strftime("%Y-%m-%d")
    df["result"] = None
    df["profit"] = None

    if os.path.exists(TRACKING_FILE):
        existing = pd.read_csv(TRACKING_FILE)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(TRACKING_FILE, index=False)
