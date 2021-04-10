import math
import os

import numpy as np
import tensorflow as tf

# see also:
# https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py

def get_pi_idx(pis, threshold):
    """ Samples the probabilities of each mixture """
    if threshold is None:
        threshold = np.random.rand(1)

    pdf = 0.0
    #  one sample, one timestep
    for idx, prob in enumerate(pis):
        pdf += prob
        if pdf > threshold:
            return idx

    #  if we get to this point, something is wrong!
    print('pdf {} thresh {}'.format(pdf, threshold))
    return idx

def set_random_params(self, stdev=0.5):
        params = self.get_weights()
        rand_params = []
        for param_i in params:
            # David's spicy initialization scheme is wild but from preliminary experiments is critical
            sampled_param = np.random.standard_cauchy(param_i.shape)*stdev / 10000.0
            rand_params.append(sampled_param) # spice things up

        self.set_weights(rand_params)


class LSTM(tf.keras.Model):
    """ create a custom LSTM cell layer for framewise processing """
    # init in Memory: mixture_dim = 32 (output_dim) * 5 (num_mix) * 3 (len([pi,mu,sigma]))
    # self.lstm = LSTM(input_dim,mixture_dim,num_timesteps,batch_size,lstm_nodes)
    def __init__(
            self,
            input_dim, # 32 (latents) + 3 (actions) = 35
            num_mdn_inputs, # 480 (mixture_dim)
            num_timesteps, # 999
            batch_size, # 100
            num_units # lstm_nodes = 256
            ):
        super(LSTM, self).__init__()

        self.input_dim = input_dim # latent_dim + num_actions = 35
        self.num_units = num_units # 256
        self.batch_size = batch_size # 100

        self.cell = tf.keras.layers.LSTMCell(
            self.num_units,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros')

        self.lstm = tf.keras.layers.RNN(
            self.cell,
            return_state=True, # return (hidden and cell) states at every timestep (= next LSTM input)
            return_sequences=True, # return output (= MDN input) at every timestep
            stateful=False)

        self.output_layer = tf.keras.layers.Dense(num_mdn_inputs)

    def get_zero_hidden_state(self, inputs):
        return [
            tf.zeros((inputs.shape[0], self.nodes)),
            tf.zeros((inputs.shape[0], self.nodes))
        ]

    def get_initial_state(self, inputs):
        return self.initial_state

    def call(self, input, state):
        self.initial_state = state
        self.lstm.get_initial_state = self.get_initial_state
        lstm_out, hidden_state, cell_state = self.lstm(input)
        output = self.output_layer(lstm_out)
        return output, hidden_state, cell_state


"""
The memory predicts how the environment will change based on the last action that has been taken.
We will use the hidden state h of the LSTM to train a Controller.
This internal representation is a compressed representation of time containing information the memory has learnt as being useful to predict the future.
"""
class Memory:
    """ combines LSTM and Gaussian Mixed Density model to generate predictions of future states based on timestep-data history """
    def __init__(
            self,
            input_dim=35, # length of latent input vector (latent code of current state observation + action values)
            latent_dim=32, # length of latent output vector (latent code of next state observation)
            num_timesteps=999, # over how many timesteps to train
            batch_size=100, # number of items in batch
            lstm_nodes=256,
            num_mixtures=5, # number of Gaussian distributed RVs to model pdf over latent output
            grad_clip=1.0,
            initial_learning_rate=0.001,
            end_learning_rate=0.00001,
            epochs=1,
            batch_per_epoch=1,
            load_model=False,
            results_dir=None
    ):
        decay_steps = epochs * batch_per_epoch
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_mixtures = num_mixtures

        # how many values to regress as input to Mixture Density network
        # 3 (pi, mu, sigma) for each mixture RV for each latent code dimension
        mixture_dim = latent_dim * num_mixtures * 3

        self.lstm = LSTM(
            input_dim,
            mixture_dim,
            num_timesteps,
            batch_size,
            lstm_nodes
        )

        # decay learning rate after decay_steps updates during training
        # until minimum of end_learning_rate is reached
        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipvalue=grad_clip)
        # dict to save and load model parameters with corresponding model names
        self.models = {
            'lstm': self.lstm
        }
        # use saved weights for models
        if load_model:
            self.load(results_dir)

    def call(self, input, state, temperature, threshold=None):
        """ forward pass to generate prediction of y at every timestep """
        input = tf.reshape(input, (1, 1, self.input_dim))
        temperature = np.array(temperature).reshape(1, 1)
        lstm_out, h_state, c_state = self.lstm(input, state)

        # input param of shape (batch_size, num_timesteps, latent_dim * num_mixtures * 3)
        # rehape into (batch_size, num_timesteps, latent_dim, num_mixtures * 3)
        output = tf.reshape(lstm_out, -1, (lstm_out.shape[0], lstm_out.shape[1], self.latent_dim, self.num_mixtures*3))

        # mixture density network layer computations
        # get pi, mu, sigma each of shape (batch_size, num_timesteps, latent_dim, num_mixtures)
        # pi, mu, sigma = tf.split(hidden_layer_output, 3, axis=3)
        pi, mu, sigma = tf.split(input, 3, axis=3)
        # softmax the pi's to ensure each distribution sums up to one
        pi = tf.keras.activations.softmax(pi)
        sigma = tf.exp(tf.maximum(sigma, 1e-8))

        #  for a single sample, single timtestep
        pi = np.array(pi).reshape(self.output_dim, pi.shape[3])
        mu = np.array(mu).reshape(self.output_dim, mu.shape[3])
        sigma = np.array(sigma).reshape(self.output_dim, sigma.shape[3])

        #  reset values every forward pass
        idxs = np.zeros(self.output_dim)
        mus = np.zeros(self.output_dim)
        sigmas = np.zeros(self.output_dim)
        # initialize latent vector (=to be predicted) with 0 values
        next_latent_pred = np.zeros(self.output_dim)

        # compute next latent code prediction based on output of lstm-mdn-processed input
        for num in range(self.output_dim):
            idx = get_pi_idx(pi[num, :], threshold=threshold)
            idxs[num] = idx
            mus[num] = mu[num, idx]
            sigmas[num] = sigma[num, idx]

            next_latent_pred[num] = mus[num] + np.random.randn() * sigmas[num] * np.sqrt(temperature)

        return next_latent_pred, h_state, c_state

    def kernel_probs(self, mu, sigma, next_latent):
        constant = 1 / math.sqrt(2 * math.pi)
        #  mu.shape
        #  (batch_size, num_timesteps, num_features, num_mix)

        #  next_latent.shape
        #  (batch_size, num_timesteps, num_features)
        #  -> (batch_size, num_timesteps, num_features, num_mixtures)
        next_latent = tf.expand_dims(next_latent, axis=-1)
        next_latent = tf.tile(next_latent, (1, 1, 1, self.num_mixtures))

        gaussian_kernel = tf.subtract(next_latent, mu)
        gaussian_kernel = tf.square(tf.divide(gaussian_kernel, sigma))
        gaussian_kernel = - 1/2 * gaussian_kernel
        conditional_probabilities = tf.divide(tf.exp(gaussian_kernel), sigma) * constant

        #  (batch_size, num_timesteps, num_features, num_mix)
        return conditional_probabilities

    def get_loss(self, lstm_out, next_latent):
        # input param of shape (batch_size, num_timesteps, latent_dim * num_mixtures * 3)
        # rehape into (batch_size, num_timesteps, latent_dim, num_mixtures * 3)
        output = tf.reshape(lstm_out, -1, (lstm_out.shape[0],lstm_out.shape[1], self.latent_dim, self.num_mixtures*3))

        # mixture density network layer computations
        # get pi, mu, sigma each of shape (batch_size, num_timesteps, latent_dim, num_mixtures)
        # pi, mu, sigma = tf.split(hidden_layer_output, 3, axis=3)
        pi, mu, sigma = tf.split(input, 3, axis=3)
        # softmax the pi's to ensure each distribution sums up to one
        pi = tf.keras.activations.softmax(pi)
        sigma = tf.exp(tf.maximum(sigma, 1e-8))

        # compute next latent code probabilities
        probs = self.kernel_probs(mu, sigma, next_latent)
        loss = tf.multiply(probs, pi)

        #  reduce along the mixes
        loss = tf.reduce_sum(loss, 3, keepdims=True)
        loss = -tf.math.log(loss)
        loss = tf.reduce_mean(loss)
        return loss

    def train_op(self, input, target, state):
        """ update network parameters via backpropagation """
        with tf.GradientTape() as tape:
            lstm_out, _, _ = self.lstm(input, state)
            loss = self.get_loss(lstm_out, target)
            gradients = tape.gradient(loss, self.lstm.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.lstm.trainable_variables))
        return loss

    def save(self, filepath):
        """ save model weights """
        filepath = os.path.join(filepath, 'models')
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        for name, model in self.models.items():
            model.save_weights('{}/{}.h5'.format(filepath, name))

    def load(self, filepath):
        """ load model weights from storage """
        filepath = os.path.join(filepath, 'models')
        print('loading model from {}'.format(filepath))

        for name, model in self.models.items():
            model.load_weights('{}/{}.h5'.format(filepath, name))
            self.models[name] = model
