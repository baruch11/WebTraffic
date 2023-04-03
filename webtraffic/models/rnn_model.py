"""RNN model."""
from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input, BatchNormalization
from webtraffic.inout import training_dataset
from webtraffic.webtraffic_utils import (SmapeLoss, create_tb_cb, smape,
                                         rename)
from sklearn.preprocessing import OneHotEncoder


@dataclass
class rnn_model:
    """RNN model."""

    dataset: training_dataset
    seq2seq: bool = True
    Nneurons: int = 30
    Nlayers: int = 1
    learning_rate: float = 5e-4
    max_delay: int = 50
    model: tf.keras.layers.Layer = field(init=False)
    epochs: int = 100
    batch_size: int = 32

    def __post_init__(self):
        """Build & compile the model."""
        output_len = self.dataset.get_forecast_horizon()
        I_std = Input(1, name="std")
        I_mean = Input(1, name="mean")
        hits, hits_1y, med, week, access, country = 1, 1, 1, 1, 4, 9
        feat_num = hits + hits_1y + med + week + access + country
        I_traffic = Input(shape=(self.max_delay, feat_num,), name="time_datas")

        x = I_traffic

        for ii in range(self.Nlayers-1):
            x = tf.keras.layers.GRU(self.Nneurons, return_sequences=True)(x)
        x = tf.keras.layers.GRU(self.Nneurons, return_sequences=self.seq2seq,
                                name="gru0")(x)

        x = BatchNormalization()(x)

        if not self.seq2seq:
            x = tf.keras.layers.Dense(output_len, name="dense0")(x)
            tensor_std = I_std
            tensor_mean = I_mean
        else:
            x = keras.layers.TimeDistributed(
                keras.layers.Dense(output_len), name="td")(x)
            tensor_std = tf.tile(tf.expand_dims(I_std, axis=1),
                                 (1, self.max_delay, output_len))
            tensor_mean = tf.tile(tf.expand_dims(I_mean, axis=1),
                                  (1, self.max_delay, output_len))

        x = tf.math.expm1(x * tensor_std + tensor_mean)

        outputs = tf.clip_by_value(x, clip_value_min=0, clip_value_max=100e6)

        self.model = tf.keras.Model(inputs=[I_traffic, I_mean, I_std],
                                    outputs=[outputs])

        @rename("smape")
        def smape_metric(y_pred, y_true):
            if self.seq2seq:
                return smape(y_pred[:, -1], y_true[:, -1])
            return smape(y_pred, y_true)

        self.model.compile(
            loss=SmapeLoss(),
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[smape_metric])

    def fit(self):
        """Fit the model."""
        es_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_smape', min_delta=0.1, patience=10,
            verbose=0, restore_best_weights=True)

        tb_cb = create_tb_cb("rnn")

        ds_train, ds_test = self._get_train_test_ds()

        self.model.fit(ds_train.shuffle(10000).batch(self.batch_size),
                       epochs=self.epochs,
                       callbacks=[tb_cb, es_cb],
                       validation_data=ds_test)

    def predict(self, X_train: pd.DataFrame):
        """Predict forecast from X_train.

        Returns
        -------
        np.array
            predictions
        """
        rnn_out = self.model.predict(self._features_preparation(X_train),
                                     batch_size=1024)
        if self.seq2seq:
            rnn_out = rnn_out[:, -1]
        ret = np.clip(rnn_out, a_min=0, a_max=None).round().astype(np.int32)
        return ret

    def _get_train_test_ds(self):
        """Return train and test td.Datasets ."""
        additional_samples = 0
        if self.seq2seq:
            additional_samples = self.max_delay
        (X_train, Y_train), val_data = self.dataset.get_training_datasets(
            additional_samples)

        ds_test = val_data
        if val_data is not None:
            ds_test = self._dataprep_tf_dataset(val_data[0], val_data[1])
            ds_test = ds_test.batch(1024)

        ds_train = self._dataprep_tf_dataset(X_train, Y_train)
        return ds_train, ds_test

    def _dataprep_tf_dataset(self, X_train, Y_train):
        """Compute data preparation and return a tf.Dataset."""
        time_datas, x_mean, x_std = self._features_preparation(X_train)
        ds_x = tf.data.Dataset.from_tensor_slices(
            {"time_datas": time_datas, "mean": x_mean, "std": x_std})
        ds_y = tf.data.Dataset.from_tensor_slices(Y_train.values)
        if self.seq2seq:
            ds_y = ds_y.map(self._to_seq2seq)

        ds = tf.data.Dataset.zip((ds_x, ds_y))
        return ds

    def _features_preparation(self, X_train: pd.DataFrame):
        """Compute features.

        Parameters
        ----------
        X_train (pd.DataFrame)
            hits, index: wiki page name, columns: dates

        Returns
        -------
        times_datas (np.array)
            1st dim: samples, 2nd dim: time steps, 3rd: features (
            current hits, 1year lag hits, access, country, day of the week)
        x_mean (np.array)
            vector of the mean of the log1p of the hits
        x_std (np.array)
            vector of the standard variance of the log1p of the hits
        """
        np_train = np.log1p(X_train.values)

        median = np.median(np_train, axis=1).reshape(-1, 1)
        median = median - np.mean(median)
        median = median / np.std(median)
        median = np.repeat(median, self.max_delay, axis=1)

        x_mean = np.mean(np_train, axis=1).reshape(-1, 1)
        x_std = (np.std(np_train, axis=1) + 1e-10).reshape(-1, 1)
        scaled_x = (np_train - x_mean) / x_std
        current_x = scaled_x[:, -self.max_delay:]

        horizon = self.dataset.get_forecast_horizon()
        lead = self.dataset.get_lead_time()
        lag1year = lead+horizon-365
        past_year = scaled_x[:, lag1year-self.max_delay:lag1year]

        weekday = pd.to_datetime(X_train.columns).weekday.\
            values[-self.max_delay:]
        weekday = (weekday - 3.) / 3.
        weekday = np.repeat(weekday.reshape(1, -1), X_train.shape[0], axis=0)

        time_datas = np.stack([current_x, weekday, median, past_year], axis=-1)

        access = self._access_onehot_encode(X_train)
        access = np.repeat(access[:, np.newaxis, :], self.max_delay, axis=1)
        time_datas = np.concatenate((time_datas, access[:, :, :]), axis=-1)

        country = self._country_encoder(X_train)
        country = np.repeat(country[:, np.newaxis, :], self.max_delay, axis=1)
        time_datas = np.concatenate((time_datas, country[:, :, :]), axis=-1)

        return time_datas, x_mean, x_std

    def _access_onehot_encode(self, X_train):
        """Return access one hot encode from X_train indexes."""
        ret = np.zeros((len(X_train), 4))
        for line, idx in enumerate(X_train.index):
            access, agent = idx.split("_")[-2:]
            encoding = 0
            if access == "mobile-web":
                encoding = 1
            if access == "desktop":
                encoding = 2
            if agent == "spider":
                encoding = 3
            ret[line, encoding] = 1
        return ret

    def _country_encoder(self, X_train):
        """Return country of the wikipedia page."""
        country = np.array([
            idx.split("_")[-3].split(".")[0]
            for idx in X_train.index]).reshape(-1, 1)

        categories = ['en', 'ja', 'de', 'fr', 'zh', 'ru', 'es', 'commons',
                      'www']
        encoder = OneHotEncoder(categories=[categories],
                                handle_unknown='error',
                                sparse_output=False)
        ret = encoder.fit_transform(country)
        return ret

    def _to_seq2seq(self, target_in: np.array):
        tgt_list = []
        out_len = self.dataset.get_forecast_horizon()
        for ii in range(self.max_delay):
            tgt_list.append(target_in[ii:ii+out_len])
        return tf.stack(tgt_list)
