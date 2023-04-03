"""Manage data loading, dataset build for training, and exports."""
import os
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

# these constants are taken from the kaggle page
TRAIN_LAST_DAY = pd.Timestamp('2017-09-10')
PRED_FIRST_DAY = pd.Timestamp('2017-09-13')
PRED_LAST_DAY = pd.Timestamp('2017-11-13')


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


@dataclass
class training_dataset:
    """Represents a training dataset with eventualy a validation dataset."""

    traffic: pd.DataFrame
    validation_set: bool = True

    def get_training_datasets(self, add_samples: int = 0):
        """Return tuples of DataFrames used to train/test the model.

        split the dataset to maximize number of samples used
        shape of _train (and test) are equal for X and Y

        if validation_set is False then the training set is built
        with the last values of self.traffic

        Parameters
        ----------
        add_samples (int)
            number of additional samples to put in the targets (Y_). Sometimes
            needed to train se2qseq models.

        Returns
        -------
        train_tuple (pd.DataFrame)
            (X_train, Y_train)
        test_tuple (pd.DataFrame)
            (X_test, Y_test) or None
        """
        horizon = self.get_forecast_horizon()
        lead = self.get_lead_time()
        Y_test = self.traffic.iloc[:, -horizon-add_samples:]
        x_last = -horizon-lead+1
        Y_train = self.traffic.iloc[:, x_last-horizon-add_samples:x_last]
        X_train = self.traffic.iloc[:, :x_last-horizon-lead+1]
        X_test = self.traffic.iloc[:, x_last-X_train.shape[1]:x_last]

        test_tuple = None
        train_tuple = (X_test, Y_test)
        if self.validation_set:
            test_tuple = (X_test, Y_test)
            train_tuple = (X_train, Y_train)

        return train_tuple, test_tuple

    def get_lead_time(self):
        """Return prediction lead time."""
        return (PRED_FIRST_DAY - TRAIN_LAST_DAY).days

    def get_forecast_horizon(self):
        """Return forecast horizon (how many days to predict)."""
        return (PRED_LAST_DAY - PRED_FIRST_DAY).days + 1


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
