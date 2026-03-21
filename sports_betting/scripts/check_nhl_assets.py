"""Quick NHL data/model/injury readiness checks.

This script validates that key NHL assets exist under the repository's
``sports_betting/data`` directory and reports actionable status messages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
NHL_HISTORICAL_PATH = DATA_ROOT / "historical" / "nhl_historical.csv"
NHL_MODEL_PATH = DATA_ROOT / "models" / "nhl_model.pkl"
NHL_INJURIES_PATH = DATA_ROOT / "injuries" / "injuries.json"


def check_nhl_data() -> bool:
    """Ensure NHL historical CSV exists."""
    if NHL_HISTORICAL_PATH.exists():
        print(f"NHL historical data found at {NHL_HISTORICAL_PATH}")
        return True

    print(f"Error: NHL historical data missing at {NHL_HISTORICAL_PATH}")
    fetch_nhl_data()
    return False


def fetch_nhl_data() -> None:
    """Placeholder hook for downloading/populating NHL historical data."""
    print("Fetching NHL historical data... (placeholder)")


def fetch_nhl_injury_data() -> pd.DataFrame:
    """Load NHL injury rows from the consolidated injury JSON file when present."""
    if not NHL_INJURIES_PATH.exists():
        return pd.DataFrame([])

    try:
        raw = pd.read_json(NHL_INJURIES_PATH)
    except ValueError:
        return pd.DataFrame([])

    if "sport" in raw.columns:
        nhl_rows = raw[raw["sport"].astype(str).str.lower() == "nhl"].copy()
    else:
        # If sport is omitted, keep rows with NHL-like team text when possible,
        # otherwise return empty to avoid misclassification.
        nhl_rows = pd.DataFrame([])

    return nhl_rows.reset_index(drop=True)


def check_nhl_injury_data() -> bool:
    """Check if NHL injury records are available in the injury source."""
    injury_data = fetch_nhl_injury_data()
    if injury_data.empty:
        print("Warning: No injury data found for NHL teams.")
        return False

    print(f"Found {len(injury_data)} NHL injury records.")
    return True


def check_nhl_model() -> bool:
    """Ensure NHL model artifact exists."""
    if NHL_MODEL_PATH.exists():
        print(f"NHL model found at {NHL_MODEL_PATH}")
        return True

    print(f"Error: NHL model file missing at {NHL_MODEL_PATH}")
    train_nhl_model()
    return False


def train_nhl_model() -> None:
    """Placeholder hook for NHL model training."""
    print("Training NHL model... (placeholder)")


def main() -> None:
    check_nhl_data()
    check_nhl_injury_data()
    check_nhl_model()


if __name__ == "__main__":
    main()
