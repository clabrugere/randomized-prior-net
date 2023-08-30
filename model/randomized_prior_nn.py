import tensorflow as tf
import numpy as np
from scipy import stats


class MLP(tf.keras.Model):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True, name="MLP"):
        super().__init__(name=name)

        self.blocks = tf.keras.Sequential(name="MLP")
        for _ in range(num_hidden - 1):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

            if batch_norm:
                self.blocks.add(tf.keras.layers.BatchNormalization())

            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.blocks.add(tf.keras.layers.Dense(dim_out))
        else:
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

    def call(self, inputs, training=None):
        return self.blocks(inputs, training=training)


class PriorNet(tf.keras.Model):
    def __init__(self, num_hidden, dim_hidden, dim_out, name="prior_net"):
        super().__init__(name=name)

        self._layers = tf.keras.Sequential()
        for _ in range(num_hidden - 1):
            self._layers.add(
                tf.keras.layers.Dense(
                    dim_hidden, activation="relu", kernel_initializer="glorot_normal", trainable=False
                )
            )

        if dim_out:
            self._layers.add(tf.keras.layers.Dense(dim_out, kernel_initializer="glorot_normal", trainable=False))
        else:
            self._layers.add(tf.keras.layers.Dense(dim_hidden, kernel_initializer="glorot_normal", trainable=False))

    def call(self, inputs):
        return self._layers(inputs)


class RandomizedPrioNet(tf.keras.Model):
    def __init__(
        self,
        num_hidden_prior=2,
        dim_hidden_prior=64,
        num_hidden_encoder=3,
        dim_hidden_encoder=64,
        dropout_encoder=0.0,
        num_subnetworks=10,
        prior_scale=1.0,
    ):
        super().__init__()

        self.prior_scale = prior_scale
        self.num_subnetworks = num_subnetworks

        self.prior_nets = []
        self.encoders = []
        for i in range(num_subnetworks):
            # non trainable prior network
            self.prior_nets.append(
                PriorNet(num_hidden=num_hidden_prior, dim_hidden=dim_hidden_prior, dim_out=1, name=f"prior_net{i}")
            )
            # trainable encoder
            self.encoders.append(
                MLP(
                    num_hidden=num_hidden_encoder,
                    dim_hidden=dim_hidden_encoder,
                    dim_out=1,
                    dropout=dropout_encoder,
                    name=f"encoder{i}",
                )
            )

    def call(self, inputs, training=None):
        out = []
        for prior, encoder in zip(self.prior_nets, self.encoders):
            prior_logits = prior(inputs)
            encoder_logits = encoder(inputs, training=training)
            out.append(encoder_logits + self.prior_scale * prior_logits)

        return tf.concat(out, axis=-1)

    def predict(self, inputs, batch_size, confidence=0.95):
        y_pred = super().predict(inputs, batch_size)
        mean = np.mean(y_pred, axis=1)
        se = np.std(y_pred, axis=1) / np.sqrt(self.num_subnetworks)
        bound = stats.t.ppf((1 + confidence) / 2, self.num_subnetworks - 1) * se

        return mean, np.stack((mean - bound, mean + bound), axis=1)
