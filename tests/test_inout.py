"""unit tests for inout functions."""
import pandas as pd
from webtraffic.inout import training_dataset, TRAIN_FIRST_DAY, TRAIN_LAST_DAY


def test_get_training_datasets():
    drange = pd.date_range(TRAIN_FIRST_DAY, TRAIN_LAST_DAY)
    traffic = pd.DataFrame(0, columns=drange.strftime("%Y-%m-%d"),
                           index=[1, 2, 3])
    tds = training_dataset(traffic)
    (X_train, Y_train), (X_test, Y_test) = tds.get_training_datasets()
    assert X_train.shape == X_test.shape
    assert Y_train.shape == Y_test.shape
    assert Y_train.shape[1] == 62
    lead_time = pd.Timedelta(3, 'd')
    x_tr_time = pd.to_datetime(X_train.columns)
    y_tr_time = pd.to_datetime(Y_train.columns)
    x_te_time = pd.to_datetime(X_test.columns)
    y_te_time = pd.to_datetime(Y_test.columns)
    assert x_tr_time[-1] + lead_time == y_tr_time[0]
    assert x_te_time[-1] + lead_time == y_te_time[0]
    assert Y_train.columns[-1] == X_test.columns[-1]
    assert X_train.columns[0] == TRAIN_FIRST_DAY
    assert Y_test.columns[-1] == TRAIN_LAST_DAY
