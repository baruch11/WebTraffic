"""Misc functions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime

from webtraffic.inout import get_root_dir

def smape_np(Mleft, Mright):
    """Return smape scores.

    Parameters:
    -----------
    Mleft, Mright (nd.array)
        the 2 tables of floats
    """
    num = 2 * np.abs(Mright - Mleft)
    denom = np.abs(Mleft) + np.abs(Mright) + np.finfo(float).eps
    return 100 / Mleft.size * np.sum(num / denom)


@dataclass
class VizualizeResults:
    """This class returns zoomed signal for easy vizualization."""

    dataset: pd.DataFrame
    page: str = 'Acier_inoxydable_fr.wikipedia.org_desktop_all-agents'
    nsamp_before: int = 50
    nsamp_after: int = 62

    def get_aligned_results(self, Y_test: pd.DataFrame, pred: np.array):
        """Return 3 aligned Series (initial datasets, targets, predictions)."""
        iloc = self.dataset.columns.get_loc(Y_test.columns[0])
        window = np.arange(-self.nsamp_before, self.nsamp_after)+iloc
        all_samples = self.dataset.loc[self.page].iloc[window]
        y_true = Y_test.loc[self.page].rename("y_true")
        y_pred = pd.DataFrame(pred,
                              index=Y_test.index,
                              columns=Y_test.columns)\
                   .loc[self.page].rename("y_pred")
        for ii in [all_samples, y_true, y_pred]:
            ii.index = pd.to_datetime(ii.index)
        return all_samples, y_true, y_pred

    def get_smape(self, Y_test: pd.DataFrame, pred: np.array):
        """Return smape score for the given page."""
        _, y_true, y_pred = self.get_aligned_results(Y_test, pred)
        return smape_np(y_true.values, y_pred.values)


def estimated_autocorrelation(x):
    """Compute autocorrelation.
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


def plot_spectrest(x, ax):
    """Compute and plot spectral estimation."""
    fft = tf.signal.rfft(x-np.mean(x))
    T = len(fft)
    ax.plot(np.abs(fft))
    ax.set_yscale("log")
    ax.grid()
    ax.set_xscale("log")
    ax.set_xticks([2*T/7., 2*T/30.5, 2*T/365.])
    ax.set_xticklabels(["weekly", "monthly", "yearly"], rotation=30)


# common tensorflow model utils


def create_tb_cb(model_name):
    """Create a tf callback for tensorboard."""
    dirname = datetime.now().strftime("%m-%d-%H-%M-%S")+"-"+model_name
    log_dir = os.path.join(get_root_dir(), "logs", dirname)
    return tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=2)


def smape(A, F):
    """Compute smape."""
    denom = tf.math.abs(A) + tf.math.abs(F) + 1e-16
    return tf.reduce_mean(2 * tf.math.abs(F - A) / denom) * 100


def smape_reg(ypred, ytrue):
    """Compute regularized smape."""
    A = tf.cast(ypred, tf.float32)
    F = tf.cast(ytrue, tf.float32)
    epsilon = 0.1
    summ = tf.maximum(tf.abs(A) + tf.abs(F) + epsilon, 0.5 + epsilon)
    return tf.reduce_mean(2 * tf.abs(A - F) / summ) * 100


class SmapeLoss(tf.keras.losses.Loss):
    """Implement smape for loss of tf models."""
    
    def __init__(self, name="smape_reg", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return smape_reg(y_pred, y_true)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
