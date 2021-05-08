from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class OneHotEncodingLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):

    def __init__(self, vocabulary=None, depth=None, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary
        indices = tf.range(len(vocabulary), dtype=tf.int64)
        table_init = tf.lookup.KeyValueTensorInitializer(vocabulary, indices)
        self.table_ohe = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets=1)
        self.depth = depth
        if depth is None:
            self.depth = self.table_ohe.size()

    def call(self,inputs):
        return  tf.one_hot(self.table_ohe.lookup(inputs), depth=tf.cast(self.depth, tf.int32))

    def get_config(self):
        return {'vocabulary': list(self.vocabulary), 'depth': int(self.depth)}


def smape(A, F):
    return tf.reduce_mean(2 * tf.math.abs(F - A) / (tf.math.abs(A) + tf.math.abs(F) + 1e-16)) * 100


def smape_reg(ypred, ytrue):
    A = tf.cast(ypred, tf.float32)
    F = tf.cast(ytrue, tf.float32)
    epsilon = 0.1
    summ = tf.maximum(tf.abs(A) + tf.abs(F) + epsilon, 0.5 + epsilon)
    return  tf.reduce_mean( 2* tf.abs(A - F) / summ) * 100


class SmapeMetric(tf.keras.losses.Loss):
    def __init__(self, name="smape", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return smape(y_pred, y_true)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class SmapeLoss(tf.keras.losses.Loss):
    def __init__(self, name="smape_reg", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return smape_reg(y_pred, y_true)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


def smape_np(A, F):
    return 100/A.size * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))


def create_tb_cb(model_name):
    return tf.keras.callbacks.TensorBoard(log_dir="./logs/"+datetime.now().strftime("%m-%d-%H-%M-%S")+"-"+model_name,
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



def get_model_inputs(df_augmented, return_seq=False):
    feat_cols = [ii for ii in df_augmented if "feat_" in ii]
    df_train = df_augmented.drop(columns=feat_cols)
    nl, nc = df_train.shape
    inputs = [tf.convert_to_tensor(df_train.index), df_train.values[:,:-62]]
    for ii in feat_cols:
        inputs.append(tf.convert_to_tensor(df_augmented[ii].values))
    output = df_train.values[:,-62:]
    if return_seq:
        output = np.empty((nl, MaxTs, 62))
        for ii in range(MaxTs):
            last = nc-MaxTs+ii+1
            output[:,ii,:] = df_train.iloc[:,last-62:last]
    return inputs, output


def plot_check_result(df_train, page, models):
    """ df: dataframe from original csv file """
    features, target = get_model_inputs(df_train.loc[[page]])
    f, vax = plt.subplots(len(models),2, figsize=(10,4*len(models)))
    fax = vax.flat

    for model in models:
        smape_score = model.evaluate(features, target, verbose=0)[1]
        pred = model.predict(features, verbose=0)[0]

        ax= next(fax)
        ax.set_title("smape: {:.2f}".format(smape_score))
        ax.plot(target.reshape(-1,), label="pred")
        ax.plot(pred.reshape(-1,), label="prediction")
        ax1= next(fax)
        ax1.plot(df_train.loc[page].values)
        ax1.plot(np.arange(0,len(pred))+len(features[1].reshape(-1,1)),pred)

    f.suptitle(page)
    plt.show()
