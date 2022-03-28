import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error

class GaussianLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.W_mu = None
        self.b_mu = None
        self.W_sigma = None
        self.b_sigma = None

    def build(self, input_shape):
        super(GaussianLayer, self).build(input_shape)

        dim = input_shape[-1]

        self.W_mu = self.add_weight(
            name='W_mu',
            shape=(dim, 1),
            initializer='glorot_normal',
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(1,),
            initializer='zeros',
            trainable=True,
        )

        self.W_sigma = self.add_weight(
            name='W_sigma',
            shape=(dim, 1),
            initializer='glorot_normal',
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(1,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, inputs):
        mu = K.dot(inputs, self.W_mu)
        mu = K.bias_add(mu, self.b_mu, data_format='channels_last')

        sigma = K.dot(inputs, self.W_sigma)
        sigma = K.bias_add(sigma, self.b_sigma, data_format='channels_last')
        sigma = K.softplus(sigma) + K.epsilon()

        return tf.squeeze(mu, axis=-1), tf.squeeze(sigma, axis=-1)


def gaussian_loss(y_true, mu, sigma):
    loss = (
            tf.math.log(sigma)
            + 0.5 * tf.math.log(2 * np.pi)
            + 0.5 * tf.square(tf.math.divide(y_true - mu, sigma))
    )
    return tf.reduce_mean(loss)


def gaussian_sample(mu, sigma):
    mu = tf.expand_dims(mu, axis=-1)
    sigma = tf.expand_dims(sigma, axis=-1)

    samples = tf.random.normal((300,), mean=mu, stddev=sigma)
    return tf.reduce_mean(samples, axis=-1)
