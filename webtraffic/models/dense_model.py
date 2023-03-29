"""Linear regression multioutput model."""
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from webtraffic.webtraffic_utils import SmapeLoss, SmapeMetric, create_tb_cb


@dataclass
class dense_model:
    """Dense regression multioutput model with smape loss."""

    output_len: int
    Ldelay: int = 100
    model: tf.keras.layers.Layer = field(init=False)
    epochs: int = 100

    def __post_init__(self):
        """Build and compile the model."""
        I_std = tf.keras.layers.Input(1, name="std")
        I_mean = tf.keras.layers.Input(1, name="mean")
        I_traffic = tf.keras.layers.Input(shape=(self.Ldelay,), name="traffic")

        x = tf.keras.layers.Dense(units=self.output_len,
                                  input_dim=self.Ldelay)(I_traffic)

        x = tf.math.expm1(x * I_std + I_mean)

        outputs = tf.clip_by_value(x, clip_value_min=0, clip_value_max=100e6)
        self.model = tf.keras.Model(inputs=[I_traffic, I_mean, I_std],
                                    outputs=[outputs])

        self.model.compile(loss=SmapeLoss(),
                           optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                           metrics=[SmapeMetric()])

    def fit(self, X_train: np.array, Y_train: np.array, val_data=None):
        """Fit the model."""
        es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_smape',
                                                 min_delta=0.1,
                                                 patience=5,
                                                 verbose=0,
                                                 restore_best_weights=True)
        tb_cb = create_tb_cb("dense")

        vd = val_data
        if val_data is not None:
            X_val, Y_val = val_data
            vd = (self._normalize_x(X_val),
                  Y_val.values)

        self.model.fit(self._normalize_x(X_train),
                       Y_train, epochs=self.epochs,
                       callbacks=[tb_cb, es_cb],
                       batch_size=32,
                       validation_data=vd)

    def _normalize_x(self, X_train):
        np_train = np.log1p(X_train.values[:, -self.Ldelay:])
        x_mean = np.mean(np_train, axis=1).reshape(-1, 1)
        x_std = (np.std(np_train, axis=1) + 1e-10).reshape(-1, 1)
        scaled_x = (np_train - x_mean) / x_std

        return [scaled_x, x_mean, x_std]

    def predict(self, X_train: np.array):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        return self.model.predict(self._normalize_x(X_train))
