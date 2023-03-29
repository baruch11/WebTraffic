
import pandas as pd
import numpy as np
from webtraffic.models.median_model import median_model
from webtraffic.inout import training_dataset


def test_median_model():
    traffic = pd.DataFrame(np.random.randint(0, 10, size=(10, 400)))
    tr_ds = training_dataset(traffic)
    model = median_model(training_dataset(traffic))
    model.fit()
    pred = model.predict(traffic.iloc[:, -50:])
    assert pred.shape[1] == tr_ds.get_forecast_horizon()
