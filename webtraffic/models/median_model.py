"""Simple model that repeats the median of the n last values."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from webtraffic.inout import training_dataset


@dataclass
class median_model:
    """Simple model that repeats the median of the n last values."""

    dataset: training_dataset
    median_depth: int = 40
    output_len: int = field(init=False)
    epochs: int = -1

    def __post_init__(self):
        """post init method."""
        self.output_len = self.dataset.get_forecast_horizon()

    def fit(self):
        """Fit the model."""
        return self

    def predict(self, X_train: pd.DataFrame):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        medians = np.median(X_train.values[:, -self.median_depth:], axis=1)
        medians = medians.reshape(-1, 1)
        medians = np.clip(medians, a_min=0, a_max=None).astype(np.int32)

        return np.repeat(medians, self.output_len, axis=1)
