"""This is a policy gradient learning training algotrithm for the gym environment cartpole"""
import logging, os
import tensorflow_probability as tfp

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import Box2D
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import numpy as np
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)

CRITIC_LR = 0.003
ACTOR_LR = 0.003
CLIP_RATIO = 0.1


class Actor(tf.keras.Model):

    def __init__(self, state_dim, action_dim, action_bound, std_bound, init=tf.keras.initializers.HeUniform()):
        super(Actor, self).__init__()
        self.opt = tf.keras.optimizers.Adam(ACTOR_LR)
        self.net = [
            Dense(32, activation='relu', kernel_initializer=init),
            Dense(32, activation='relu', kernel_initializer=init),
        ]
        self.mu = Dense(action_dim, activation='tanh')
        self.mu_bound = tf.keras.layers.Lambda(lambda x: x * self.action_bound)
        self.sigma = tf.keras.layers.Dense(action_dim)

class PPO(tf.keras.Model):

    def __init__(self, action_bound=2, num_actions=4, hdim1=32, hdim2=32, init=tf.keras.initializers.HeUniform()):
        super(PPO, self).__init__()
        self.action_bound = action_bound
        self.opt_a = tf.keras.optimizers.Adam(CRITIC_LR)
        self.opt_c = tf.keras.optimizers.Adam(ACTOR_LR)

        self.critic_net = [
            tf.keras.layers.Dense(hdim1, activation=tf.nn.relu, kernel_initializer=init, name="critic1"),
            tf.keras.layers.Dense(hdim2, activation=tf.nn.relu, kernel_initializer=init, name="critic2"),
            tf.keras.layers.Dense(1, kernel_initializer=init, name="critic_out")
        ]
        self.actor_net = [
            tf.keras.layers.Dense(hdim1, activation=tf.nn.relu, kernel_initializer=init, name="actor1"),
            tf.keras.layers.Dense(hdim2, activation=tf.nn.relu, kernel_initializer=init, name="actor2")
        ]
        self.action_values_mu = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.tanh,
                                                      kernel_initializer=init, name="actor_mu")
        self.action_bound = tf.keras.layers.Lambda(lambda x: x * self.action_bound)
        self.action_values_logsigma = tf.keras.layers.Dense(num_actions, kernel_initializer=init, name="actor_sigma",
                                                            activation='softplus')

    @tf.function
    def call(self, x):
        output = {}
        v = x
        for layer in self.critic_net:
            v = layer(v)
        # later filter model.trainable_variables according to the layers' names to compute and apply gradients separately
        # compute the state values again when optimizing because returns from manager cannot be backpropagated
        for layer in self.actor_net:
            x = layer(x)
        mus = self.action_values_mu(x)
        #mus = self.action_bound(mus)
        sigmas = self.action_values_logsigma(x)
        sigmas = tf.clip_by_value(sigmas, -20, 2)

        output["value_estimate"] = v
        output["mu"] = mus
        output["sigma"] = tf.exp(sigmas)

        return output

    def critic_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def critic_train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt_c.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def actor_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-CLIP_RATIO, 1.0+CLIP_RATIO)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def actor_train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt_a.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
