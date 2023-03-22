"""Linear regression multioutput model."""
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from webtraffic.webtraffic_utils import SmapeLoss, SmapeMetric, create_tb_cb


@dataclass
class linear_model:
    """Linear regression multioutput model with smape loss."""

    input_shape: int
    output_len: int
    Ldelay: int = 100
    model: tf.keras.layers.Layer = field(init=False)
    epochs: int = 100

    def __post_init__(self):
        """Build and compile the model."""
        traffic = tf.keras.layers.Input(shape=(self.input_shape,))
        traffic_lim = traffic[:, -self.Ldelay:]
        outputs = tf.keras.layers.Dense(units=self.output_len,
                                        input_dim=self.Ldelay)(traffic_lim)

        self.model = tf.keras.Model(inputs=[traffic], outputs=[outputs])

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
        tb_cb = create_tb_cb("linear")

        self.model.fit(X_train, Y_train, epochs=self.epochs,
                       callbacks=[tb_cb, es_cb],
                       batch_size=32,
                       validation_data=val_data)

    def predict(self, X_train: np.array):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        return self.model.predict(X_train)
