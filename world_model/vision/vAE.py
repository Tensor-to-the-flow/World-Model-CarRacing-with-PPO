import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


class VarEncoder(Model):

    def __init__(self, embedding_size = 10):
        super(VarEncoder, self).__init__()

        self.encoder = [
                tf.keras.layers.Conv2D(filters = 32, kernel_size = 4, strides = 2, padding = 'valid', activation = 'relu'),
                tf.keras.layers.Conv2D(filters = 64, kernel_size = 4, strides = 2, padding = 'valid', activation = 'relu'),
                tf.keras.layers.Conv2D(filters = 128, kernel_size = 4, strides = 2, padding = 'valid', activation = 'relu'),
                tf.keras.layers.Conv2D(filters = 256, kernel_size = 4, strides = 2, padding = 'valid', activation = 'relu'),


                tf.keras.layers.Flatten(),
            ]

    def call(self, x):
        for layer in self.encoder:
            x = layer(x)
        # Define the multivariate normal distribution
        # using the first half of the output of the dense layer as mean and the second half as variance
        x = tfp.distributions.MultivariateNormalDiag(loc=x[:,:self.embedding_size], scale_diag=x[:,self.embedding_size:])
        return x


class VarDecoder(Model):
    def __init__(self):
        super(VarDecoder, self).__init__()

        self.decoder = [
                # The input shape for the first decoding layer has again the shape
                # of the embedding size, because we sample one embedding value
                # for each (µ and σ) pair
                tf.keras.layers.InputLayer(input_shape=(embedding_size)),
                tf.keras.layers.Dense(units=7*7*64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu),
	            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu),
		        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu),
		        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
            ]

    def call(self, x):
        for layer in self.my_layers:
            x = layer(x)
        return x


class VarAutoencoder(Model):
    def __init__(self, prior, embedding_size = 10):
        super(VarAutoencoder, self).__init__()
        self.prior = prior
        self.input_layer = tf.keras.layers.Input(my_input_shape)

        self.encoder = VarEncoder(embedding_size)

        self.decoder = VarDecoder(embedding_size)

        self.out = self.call(self.input_layer)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x.sample())
        return x



def compute_loss(self):
	logits_flat = tf.layers.flatten(self.reconstructions)
	labels_flat = tf.layers.flatten(self.resized_image)
	reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
	kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
	vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
	return vae_loss




def train_model(model, train_dataset, test_dataset, num_epochs, loss_function, optimizer):
    running_average_factor = 0.95

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []

    # Testing model performance on train and test data before learning
    train_loss = test(model, train_dataset, loss_function)
    train_losses.append(train_loss)

    test_loss = test(model, test_dataset, loss_function)
    test_losses.append(test_loss)

    # Display loss and accuracy before training
    print('Starting loss:')
    print('Train loss: ',train_loss)
    print('Test loss: ',test_loss)

    # Train loop for num_epochs epochs.
    for epoch in range(num_epochs):
        # Training
        running_average_loss = 0
        for model_input in train_dataset:
            train_loss = train_step(model, model_input, loss_function, optimizer)
            running_average_loss = running_average_factor * running_average_loss  + (1 - running_average_factor) * train_loss

        train_losses.append(running_average_loss.numpy())

        # Testing
        test_loss = test(model, test_dataset, loss_function)
        test_losses.append(test_loss)

        # Display loss and accuracy for current epoch
        print('Train loss: ',running_average_loss.numpy())
        print('Test loss: ',test_loss)

    model_performance = {
        "train_loss": train_losses,
        "test_loss": test_losses
    }
    return model_performance
