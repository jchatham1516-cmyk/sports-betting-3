import os
import pickle
from sklearn.linear_model import LogisticRegression

def build_payload(sport: str) -> dict:
    return {
        "sport": sport,
        "moneyline_models": LogisticRegression(),
        "spread_models": LogisticRegression(),
        "total_models": LogisticRegression(),
    }

def main() -> None:
    os.makedirs("sports_betting/data/models", exist_ok=True)

    for sport in ["nba", "nfl", "nhl"]:
        path = f"sports_betting/data/models/{sport}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(build_payload(sport), f)

    print("Placeholder model artifacts created successfully.")

if __name__ == "__main__":
    main()
