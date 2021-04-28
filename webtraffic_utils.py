import tensorflow as tf

from tensorflow.keras import layers


class OneHotEncodingLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):

    def __init__(self, vocabulary=None, depth=None, minimum=None):
        super().__init__()
        self.vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)  

        if vocabulary:
            self.vectorization.set_vocabulary(vocabulary)
        self.depth = depth   
        self.minimum = minimum

    def adapt(self, data):
        self.vectorization.adapt(data)
        vocab = self.vectorization.get_vocabulary()
        self.depth = len(vocab)
        indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
        self.minimum = min(indices)

    def call(self,inputs):
        vectorized = self.vectorization.call(inputs)
        subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
        encoded = tf.one_hot(subtracted, self.depth)
        return layers.Reshape((self.depth,))(encoded)

    def get_config(self):
        return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}


def smape(A, F):
    return tf.reduce_mean(2 * tf.math.abs(F - A) / (tf.math.abs(A) + tf.math.abs(F) + 1e-16)) * 100 

def smape_reg(A, F):
    epsilon = 0.1
    summ = tf.maximum(tf.abs(A) + tf.abs(F) + epsilon, 0.5 + epsilon)
    return tf.abs(A - F) / summ * 2.0 * 100

def smape_np(A, F):
    return 100/A.size * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
