"""Simple model that repeats the median of the n last values."""
import numpy as np

from dataclasses import dataclass


@dataclass
class median_model:
    """Simple model that repeats the median of the n last values."""

    median_depth: int = 40
    num_pred: int = -1

    def fit(self, X_train: np.array, Y_train: np.array):
        """Fit the model."""
        self.num_pred = Y_train.shape[1]
        return self

    def predict(self, X_train: np.array):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        medians = np.median(X_train[:, -self.median_depth:], axis=1)
        medians = medians.reshape(-1, 1)
        medians = np.clip(medians, a_min=0, a_max=None).astype(np.int32)

        return np.repeat(medians, self.num_pred, axis=1)