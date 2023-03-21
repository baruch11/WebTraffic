from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras



# rnn
class OneHotEncodingLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    """ one hot encoding layer """
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


class preprocessing_rnn(tf.keras.layers.Layer):
    """ layer in charge of the inputs of the rnn """
    def __init__(self, max_delay=100, use_metadata=False, use_past_year= False, **kwargs):
        super().__init__(**kwargs)
        self.max_delay = max_delay
        self.use_metadata = use_metadata
        self.use_past_year = use_past_year

    def call(self, inputs, access1h):
        ret = inputs[:,-self.max_delay:, np.newaxis]
        output_len = 62
        if self.use_metadata:
            access_broadcast = tf.tile(access1h[:,np.newaxis,:],[1, self.max_delay, 1])
            ret = tf.concat([ret, access_broadcast], axis=2)

        if self.use_past_year:
            past_year = inputs[:, -self.max_delay - 365 + output_len: -365 + output_len, np.newaxis]
            ret = tf.concat([ret, past_year], axis=2)
        return ret

    def get_config(self):
        return {'max_delay': int(self.max_delay), 'use_metadata': self.use_metadata, 'use_past_year': self.use_past_year}


def get_rnn_model(_Seq2seq, Nneurons=20, Nlayers=1, max_delay=50, use_past_year=False, use_metadata=False):

    output_len = 62
    I_page = tf.keras.layers.Input(shape=(), dtype=object)
    I_traffic = tf.keras.layers.Input(shape=(max_delay,))

    voc_access = ['all-access_all-agents', 'all-access_spider', 'desktop_all-agents',
                  'mobile-web_all-agents']
    access1h = OneHotEncodingLayer(voc_access, name="ohAccess")(I_page)

    factors = tf.reduce_max(I_traffic, axis=1, keepdims=True)
    x = tf.divide(I_traffic, factors + 1e-10)

    x = preprocessing_rnn(max_delay, use_metadata, use_past_year)(x, access1h)
    for ii in range(Nlayers-1):
        x = tf.keras.layers.GRU(Nneurons, return_sequences=True)(x)
    x = tf.keras.layers.GRU(Nneurons, return_sequences=_Seq2seq, name="gru0")(x)

    if not _Seq2seq:
        x= tf.keras.layers.Dense(output_len, name="dense0")(x)
    else:
        x = keras.layers.TimeDistributed(keras.layers.Dense(output_len), name="td")(x)
        factors = tf.cast(tf.tile(tf.expand_dims(factors, axis=1), (1, max_delay, output_len)), tf.float32)

    outputs = tf.multiply(x, factors)

    model_rnn = tf.keras.Model(inputs=[I_page, I_traffic], outputs=[outputs])

    model_rnn.compile(loss=SmapeLoss(), optimizer=tf.optimizers.Adam(learning_rate=1e-3), metrics=[SmapeMetric()])
    return model_rnn



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



def get_model_inputs(df_augmented, return_seq=0):
    feat_cols = [ii for ii in df_augmented if "feat_" in ii]
    df_train = df_augmented.drop(columns=feat_cols)
    nl, nc = df_train.shape
    inputs = [tf.convert_to_tensor(df_train.index), df_train.values[:,:-62]]
    for ii in feat_cols:
        inputs.append(tf.convert_to_tensor(df_augmented[ii].values))
    output = df_train.values[:,-62:]
    if return_seq>0:
        output = np.empty((nl, return_seq, 62))
        for ii in range(return_seq):
            last = nc-return_seq+ii+1
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


def make_seq2seq_dataset(df):
    """ seq2seq training/test as tf.dataset in order to save memory """
    bsz, nsmp = df.shape
    rnn_delay = nsmp-62
    ds = tf.data.Dataset.from_tensor_slices((df.index, df))

    def target_seq2seq(index, df):
        #target = tf.zeros((bsz, rnn_delay, 62))
        tgt_list = []
        for ii in range(rnn_delay):
            last = 62+ii+1
            tgt_list.append(df[last-62:last])
        return (index, df[:-62]), tf.stack(tgt_list)

    return ds.map(target_seq2seq)
