"""Manage data loading, dataset build for training, and exports."""
import os
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

TRAIN_FIRST_DAY = '2015-07-01'
TRAIN_LAST_DAY = '2017-09-10'
PRED_FIRST_DAY = '2017-09-13'
PRED_LAST_DAY = '2017-11-13'


def get_root_dir() -> str:
    """Return project root directory path."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_dir, "..")


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
    datapath = "data/web-traffic-time-series-forecasting/train_2.csv.zip"
    traffic_ds = pd.read_csv(os.path.join(get_root_dir(), datapath))
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


@dataclass
class Submission:
    """This class represents the kaggle submission.

    After exporting in a csv file (let's say subm_median.csv), submit
    the result with the following command:
      kaggle competitions submit -f subm_med.csv -m 'median model' web-traffic-time-series-forecasting
    """

    keys_path: str
    predictions: np.array
    index: pd.Index
    start_date: str = PRED_FIRST_DAY
    end_date: str = PRED_LAST_DAY

    def to_csv(self, output_path: str):
        """Write a csv file with the kaggle submission."""
        logging.info("Loading output keys")
        submission = pd.read_csv(self.keys_path).set_index("Page")
        logging.info("Exporting submission in %s", output_path)
        predictions = self._serialize_predictions()
        submission.loc[predictions.index, "Visits"] = predictions
        submission.to_csv(output_path, encoding='utf-8', index=False)

    def _serialize_predictions(self):
        """Return a pd.Series with predictions indexed like key2.csv."""
        out_date = pd.date_range(start=self.start_date,
                                 end=self.end_date,
                                 freq="1D").strftime("%Y-%m-%d").to_list()
        ret = pd.DataFrame(self.predictions,
                           columns=out_date,
                           index=self.index).stack().rename("Visits")
        ret.index = [ii[0]+"_"+ii[1] for ii in ret.index]
        return ret
