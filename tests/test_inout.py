"""unit tests for inout functions."""
import pandas as pd

from webtraffic.inout import get_training_datasets


def test_get_training_datasets():
    traffic = pd.DataFrame(
        0, columns=list(range(500)), index=[1, 2, 3])
    X_train, X_test, Y_train, Y_test = get_training_datasets(traffic)
    assert X_train.shape == X_test.shape
    assert Y_train.shape == Y_test.shape
    assert Y_train.shape[1] == 62
    assert X_train.columns[-1] + 3 == Y_train.columns[0]
    assert X_test.columns[-1] + 3 == Y_test.columns[0]
    assert Y_train.columns[-1] == X_test.columns[-1]
    assert X_train.columns[0] == 0
    assert Y_test.columns[-1] == traffic.shape[1]-1
