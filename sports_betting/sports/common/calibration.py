"""Probability calibration wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Simple isotonic calibrator with clipping."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        self.model.fit(probs, y_true)
        self.fitted = True

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.clip(probs, 0.01, 0.99)
        return np.clip(self.model.predict(probs), 0.01, 0.99)
