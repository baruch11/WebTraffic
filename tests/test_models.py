
import pandas as pd
import numpy as np
from webtraffic.models.median_model import median_model


def test_median_model():
    X_train = np.random.randint(0, 10, size=(10, 400))
    Y_train = np.random.randint(0, 10, size=(10, 5))
    model = median_model()
    model.fit(X_train, Y_train)
    pred = model.predict(X_train)
    assert pred.shape == Y_train.shape
