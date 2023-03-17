"""Manage data loading, dataset build for training, and exports."""
import os
import pandas as pd
import numpy as np

TRAIN_FIRST_DAY = '2015-07-01'
TRAIN_LAST_DAY = '2017-09-10'
PRED_FIRST_DAY = '2017-09-13'
PRED_LAST_DAY = '2017-11-13'


def load_data(nsamples=None):
    """Return data input dataframe.

    Parameters
    ----------
    nsamples (int)
        Nb samples max to load (if None load all samples)

    Returns
    -------
    traffic_ds : (pd.DataFrame)
        index: site, columns: days, data: page hits
    """
    this_dir = os.path.dirname(os.path.realpath(__file__))
    datapath = "data/web-traffic-time-series-forecasting/train_2.csv.zip"
    traffic_ds = pd.read_csv(os.path.join(this_dir, "..", datapath))
    traffic_ds = traffic_ds.set_index("Page").fillna(0).astype(np.int32)
    if nsamples is not None:
        traffic_ds = traffic_ds.sample(nsamples)
    return traffic_ds


def get_training_datasets(traffic):
    """Return tuple of DataFrames used to train/test the model.

    split the dataset to maximize number of samples used
    shape of _train (and test) are equal for X and Y

    Returns:
    --------
    X_train, X_test, Y_train, Y_test (pd.DataFrame)
    """
    horizon = _get_forecast_horizon()
    lead = _get_lead_time()
    Y_test = traffic.iloc[:, -horizon:]
    x_last = -horizon-lead+1
    Y_train = traffic.iloc[:, x_last-horizon:x_last]
    X_train = traffic.iloc[:, :x_last-horizon-lead+1]
    X_test = traffic.iloc[:, x_last-X_train.shape[1]:x_last]

    return X_train, X_test, Y_train, Y_test


def _get_lead_time():
    """Return prediction lead time."""
    return (pd.Timestamp(PRED_FIRST_DAY) -
            pd.Timestamp(TRAIN_LAST_DAY)).days


def _get_forecast_horizon():
    """Return forecast horizon (how many days to predict)."""
    return (pd.Timestamp(PRED_LAST_DAY) -
            pd.Timestamp(PRED_FIRST_DAY)).days + 1
