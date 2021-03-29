import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Conv2DTranspose, Conv2D, Flatten
import os


class VarAE(tf.keras.Model):
    """This Variational Autoencoder's parameters are based on the world model paper by Schmidhuber """
    def __init__(self, latent_dim=32):
        super(VarAE, self).__init__()
        self.latent_dim = latent_dim

        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        self.encoder = tf.keras.Sequential([
                Conv2D(filters=32, kernel_size=4, strides=2, padding='valid', activation='relu'),
                Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu'),
                Conv2D(filters=128, kernel_size=4, strides=2, padding='valid', activation='relu'),
                Conv2D(filters=256, kernel_size=4, strides=2, padding='valid', activation='relu'),
                Flatten(),
                Dense(latent_dim+latent_dim)

            ])
        self.decoder = tf.keras.Sequential([
                # The input shape for the first decoding layer has again the shape
                # of the embedding size, because we sample one embedding value
                # for each (µ and σ) pair
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=4*256, activation=tf.nn.relu),
                Reshape(target_shape=([-1, 1, 4*256])),
                Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu),
                Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu),
                Conv2DTranspose(filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu),
                Conv2DTranspose(filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
            ])

        self.models = {
            'encoder': self.encoder,
            'decoder': self.decoder
        }

    @tf.function
    def call(self, batch):

        means, logvars = tf.split(
            self.encoder(batch), num_or_size_splits=2, axis=1)

        epsilon = tf.random.normal(shape=means.shape)
        latent = means + epsilon * tf.exp(logvars * 0.5)

        generated = self.decoder(latent)

        return generated

    def loss(self, batch):
        kl_tolerance = 0.5

        means, logvars = tf.split(
            self.encoder(batch), num_or_size_splits=2, axis=1)

        epsilon = tf.random.normal(shape=means.shape)
        latent = means + epsilon * tf.exp(logvars * 0.5)

        generated = self.decoder(latent)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(batch - generated), axis=[1, 2, 3])
        )

        unclipped_kl_loss = - 0.5 * tf.reduce_sum(
            1 + logvars - tf.square(means) - tf.exp(logvars),
            axis=1
        )

        kl_loss = tf.reduce_mean(
            tf.maximum(unclipped_kl_loss, kl_tolerance * self.latent_dim)
        )
        return {
            'reconstruction-loss': reconstruction_loss,
            'unclipped-kl-loss': unclipped_kl_loss,
            'kl-loss': kl_loss
        }

    def backward(self, batch):
        with tf.GradientTape() as tape:
            losses = self.loss(batch)
            gradients = tape.gradient(
                sum(losses.values()), self.trainable_variables
            )

        self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
        )
        return losses

    def save(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath, 'models')
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))

        for name, model in self.models.items():
            model.save_weights('{}/{}.h5'.format(filepath, name))

    def load(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath, 'models')
        print('loading model from {}'.format(filepath))

        for name, model in self.models.items():
            model.load_weights('{}/{}.h5'.format(filepath, name))
            self.models[name] = model
