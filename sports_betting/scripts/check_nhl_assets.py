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
DEFAULT_FILTER_THRESHOLDS = {
    "min_edge": 5e-05,
    "min_ev": 5e-05,
    "min_confidence": 0.1,
}


def check_nhl_data() -> bool:
    """Ensure NHL historical CSV exists."""
    if NHL_HISTORICAL_PATH.exists():
        print(f"NHL historical data found at {NHL_HISTORICAL_PATH}")
        return True

    print(f"Error: NHL historical data missing at {NHL_HISTORICAL_PATH}")
    fetch_nhl_data()
    return False


def fetch_nhl_data() -> None:
    """Create a minimal NHL historical CSV placeholder when none exists."""
    print("Fetching NHL historical data... (local placeholder)")
    NHL_HISTORICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    placeholder = pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "moneyline_home",
            "moneyline_away",
        ]
    )
    placeholder.to_csv(NHL_HISTORICAL_PATH, index=False)
    print(f"Saved placeholder NHL historical data to {NHL_HISTORICAL_PATH}")


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


def adjust_filter_thresholds(
    min_edge: float = DEFAULT_FILTER_THRESHOLDS["min_edge"],
    min_ev: float = DEFAULT_FILTER_THRESHOLDS["min_ev"],
    min_confidence: float = DEFAULT_FILTER_THRESHOLDS["min_confidence"],
) -> dict[str, float]:
    """Return low NHL filter thresholds intended to allow more candidate bets."""
    thresholds = {
        "min_edge": float(min_edge),
        "min_ev": float(min_ev),
        "min_confidence": float(min_confidence),
    }
    print(
        "Adjusting filters: "
        f"min_edge={thresholds['min_edge']}, "
        f"min_ev={thresholds['min_ev']}, "
        f"min_confidence={thresholds['min_confidence']}"
    )
    return thresholds


def apply_injury_impact() -> pd.DataFrame:
    """Load NHL injury rows so downstream prediction code can merge injury impact."""
    injury_data = fetch_nhl_injury_data()
    if injury_data.empty:
        print("No injury data found.")
        return injury_data
    print(f"Found {len(injury_data)} injury records.")
    return injury_data


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
    adjust_filter_thresholds()
    check_nhl_injury_data()
    apply_injury_impact()
    check_nhl_model()


if __name__ == "__main__":
    main()
