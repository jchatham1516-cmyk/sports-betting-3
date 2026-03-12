"""Base sport model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from sports_betting.models.entities import Prediction


class SportModel(ABC):
    sport: str

    @abstractmethod
    def train(self, historical_df: pd.DataFrame) -> None:
        """Train model from historical feature set."""

    @abstractmethod
    def predict_daily(self, daily_df: pd.DataFrame) -> list[Prediction]:
        """Generate market predictions."""
