from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


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
                                          histogram_freq=10
                                         )

def plot_check_result(x_check, predict_func, ax):
    """
    Args:
        x_check: np.array
    """
    pred = predict_func(x_check[:-output_len])
    ax.plot(x_check)
    ax.plot(np.arange(output_len)+len(x_check)-output_len, pred)
