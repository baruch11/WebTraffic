from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class OneHotEncodingLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):

    def __init__(self, vocabulary=None, depth=None):
        super().__init__()
        self.vocabulary = vocabulary
        indices = tf.range(len(vocabulary), dtype=tf.int64)
        table_init = tf.lookup.KeyValueTensorInitializer(vocabulary, indices)
        self.table_ohe = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets=1)
        self.depth = depth
        if depth is None:
            self.depth = tf.cast(self.table_ohe.size(), tf.int32)

    def call(self,inputs):
        return  tf.one_hot(self.table_ohe.lookup(inputs), depth=self.depth)

    def get_config(self):
        return {'vocabulary': self.vocabulary, 'depth': self.depth}


def smape(A, F):
    return tf.reduce_mean(2 * tf.math.abs(F - A) / (tf.math.abs(A) + tf.math.abs(F) + 1e-16)) * 100


def smape_reg(A, F):
    epsilon = 0.1
    summ = tf.maximum(tf.abs(A) + tf.abs(F) + epsilon, 0.5 + epsilon)
    return tf.abs(A - F) / summ * 2.0 * 100


def smape_np(A, F):
    return 100/A.size * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))


def create_tb_cb(model_name):
    return tf.keras.callbacks.TensorBoard(log_dir="./logs/"+model_name+"-"+datetime.now().strftime("%H-%M-%S"),
                                          histogram_freq=10)


def estimated_autocorrelation(x):
    """
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
    fft = tf.signal.rfft(x-np.mean(x))
    T = len(fft)
    ax.plot(np.abs(fft))
    ax.set_yscale("log")
    ax.grid()
    ax.set_xscale("log")
    ax.set_xticks([2*T/7., 2*T/30.5, 2*T/365.])
    ax.set_xticklabels(["weekly", "monthly", "yearly"], rotation=30)


def ds_from_dataframe(df):
    output_len = 62
    page = df["Page"]
    nums = df.drop(columns="Page").fillna(0)
    feature = nums.values[:, :-output_len]
    target = nums.values[:, -output_len:]
    return tf.data.Dataset.from_tensor_slices(((page,feature),target))



def plot_check_result(df, page, model):
    """ df: dataframe from original csv file """
    lds = ds_from_dataframe(df.loc[df["Page"] == page])
    smape_score = model.evaluate(lds.batch(1), verbose=0)[1]

    features, target = list(lds.take(1))[0]
    pred = model.predict(lds.batch(1), verbose=0)[0]

    f, (ax, ax1) = plt.subplots(1,2, figsize=(10,4))
    ax.set_title("smape: {:.2f}".format(smape_score))
    ax.plot(target, label="pred")
    ax.plot(pred, label="prediction")
    ax1.plot(np.r_[features[1],target])
    ax1.plot(np.arange(0,len(pred))+len(features[1]),pred)

    f.suptitle(page)
    plt.show()
