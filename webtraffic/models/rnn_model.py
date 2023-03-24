"""RNN model."""
from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input

from webtraffic.webtraffic_utils import SmapeMetric, SmapeLoss, create_tb_cb


@dataclass
class rnn_model:
    """RNN model."""

    input_shape: int
    output_len: int
    seq2seq: bool = False
    Nneurons: int = 20
    Nlayers: int = 1
    max_delay: int = 50
    Lmedian: int = 40
    model: tf.keras.layers.Layer = field(init=False)
    epochs: int = 100
    mean: np.array = field(init=False)
    std: np.array = field(init=False)

    def __post_init__(self):
        """Build & compile the model."""
        I_median = Input(1,)
        I_traffic = tf.keras.layers.Input(shape=(self.input_shape,))

        traffic_lim = I_traffic[:, -self.max_delay:]

        x = traffic_lim[:, :, np.newaxis]

        for ii in range(self.Nlayers-1):
            x = tf.keras.layers.GRU(self.Nneurons, return_sequences=True)(x)
        x = tf.keras.layers.GRU(self.Nneurons, return_sequences=self.seq2seq,
                                name="gru0")(x)

        x = tf.concat([x, I_median], axis=1)

        if not self.seq2seq:
            outputs = tf.keras.layers.Dense(self.output_len, name="dense0")(x)
        else:
            outputs = keras.layers.TimeDistributed(
                keras.layers.Dense(self.output_len), name="td")(x)

        self.model = tf.keras.Model(inputs=[I_traffic, I_median],
                                    outputs=[outputs])

        self.model.compile(loss=SmapeLoss(),
                           optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                           metrics=[SmapeMetric()])

    def fit(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, val_data=None):
        """Fit the model."""
        es_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_smape', min_delta=0.1, patience=10,
            verbose=0, restore_best_weights=True)

        tb_cb = create_tb_cb("rnn")

        np_train = X_train.values
        median = np.median(np_train[:, -self.Lmedian:], axis=1)
        self.mean = np.mean(np_train, axis=1).reshape(-1, 1)
        self.std = np.std(np_train - self.mean).reshape(-1, 1) + 1e-10

        vd = val_data
        if val_data is not None:
            X_val, Y_val = val_data
            vd = ([(X_val.values - self.mean) / self.std,
                   np.median(X_val.values[:, -self.Lmedian:], axis=1)],
                  Y_val.values)

        self.model.fit([(np_train - self.mean) / self.std, median],
                       Y_train.values,
                       epochs=self.epochs,
                       callbacks=[tb_cb, es_cb],
                       batch_size=32,
                       validation_data=vd)

    def predict(self, X_train: pd.DataFrame):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        median = np.median(X_train.values[:, -self.Lmedian:], axis=1)
        X_scaled = (X_train.values - self.mean) / self.std
        rnn_out = self.model.predict([X_scaled, median])
        ret = np.clip(rnn_out, a_min=0, a_max=None).astype(np.int32)
        return ret


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
        return  tf.one_hot(self.table_ohe.lookup(inputs),
                           depth=tf.cast(self.depth, tf.int32))

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
