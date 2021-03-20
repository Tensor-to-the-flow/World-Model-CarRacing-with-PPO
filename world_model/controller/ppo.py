"""This is a policy gradient learning training algotrithm for the gym environment cartpole"""
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)


class PPO(tf.keras.Model):

    def __init__(self, n_actions=2, lsize1=64, lsize2=64):
        super(PPO, self).__init__()

        self.net = [
            tf.keras.layers.Dense(lsize1, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(lsize2, activation=tf.nn.leaky_relu)
        ]
        self.mu = tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)
        self.sigma = tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)

    @tf.function
    def call(self, x):
        result = {}

        for layer in self.net:
            x = layer(x)

        result["mu"] = self.mu(x)
        result["sigma"] = self.sigma(x)
        return result

    def reparameterize(self, mean, logvar):
        e = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(logvar * .5) * e
