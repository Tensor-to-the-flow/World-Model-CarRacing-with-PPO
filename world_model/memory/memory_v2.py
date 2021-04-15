# install mdn layer implementation
# !pip install keras-mdn-layer
import numpy as np
# import the usual ML/ANN toolsets
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions
import mdn # imports the MDN layer
import random
import os


class Memory(tf.keras.Model):
    """ Memory class to model sequential data of arbitrary dimension using a Gaussian Mixture """
    def __init__(
            self,
            input_dim=35,
            lstm_nodes=256,
            output_dim=32,
            num_mixtures=5,
            num_timesteps=999,
            hidden_units=None,
            batch_size=100,
            grad_clip=1.0,
            initial_learning_rate=0.001,
            end_learning_rate=0.00001,
            epochs=1,
            batch_per_epoch=1,
            load_model=False,
            results_dir=None,
        ):
        super(Memory, self).__init__(name="Memory")

        decay_steps = epochs * batch_per_epoch
        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate
        )

        self.optimizer = tf.keras.optimizers.Adam(0.001, clipvalue=grad_clip)
        self.loss_function = mdn.get_mixture_loss_func(output_dim, num_mixtures)

        self.num_timesteps = num_timesteps
        self.lstm_nodes = lstm_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures

        #lstm_cell = tf.keras.layers.LSTMCell(lstm_nodes, kernel_initializer='glorot_uniform',recurrent_initializer='glorot_uniform',bias_initializer='zeros',name='lstm_cell')

        self.lstm = tf.keras.layers.LSTM(lstm_nodes, return_sequences=True, return_state=True, name='lstm_layer')

        if hidden_units is None:
            self.hidden_layers = []
        else:
            self.hidden_layers = [tf.keras.layers.Dense(n_units, activation='relu') for n_units in hidden_units]

        self.mdn_out = tf.keras.layers.TimeDistributed(mdn.MDN(output_dim,num_mixtures, name='mdn_outputs'), name='td_mdn')

        self.components = {
            'lstm': self.lstm,
            'gaussian-mix': self.mdn_out,
            'hidden_layers': self.hidden_layers
        }

        if load_model:
            self.load(results_dir)

    def save(self, filepath):
        """ save model weights """
        filepath = os.path.join(filepath, 'models')
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        for name, component in self.components.items():
            component.save_weights('{}/{}.h5'.format(filepath, name))

    def load(self, filepath):
        """ load model weights """
        filepath = os.path.join(filepath, 'models')
        print('loading model from {}'.format(filepath))

        for name, component in self.components.items():
            component.load_weights('{}/{}.h5'.format(filepath, name))
            self.components[name] = component

    @tf.function
    def get_zero_hidden_state(self, inputs):
        return [tf.zeros((inputs.shape[0], self.lstm_nodes), dtype=tf.dtypes.float64),
                tf.zeros((inputs.shape[0], self.lstm_nodes), dtype=tf.dtypes.float64)]

    @tf.function
    def get_initial_state(self, inputs):
        return self.initial_state

    def get_y_pred(self, mix_params, temperature=1.0):
        # use sampling function provided by the mdn module to generate predictions
        y_samples = np.apply_along_axis(
            mdn.sample_from_output, 2, mix_params, self.output_dim,
            self.num_mixtures, temp=temperature)
        return y_samples

    # only eager execution possible currently
    #@tf.function
    def call(self, inputs, state, temperature=1.15):
        # inputs shape=(batch_size,num_timesteps=1,input_dim)
        # single timestep processing per call with input
        inputs = tf.reshape(inputs, (inputs.shape[0], self.num_timesteps, self.input_dim))

        self.initial_state = state
        self.lstm.get_initial_state = self.get_initial_state
        lstm_out, h_state, c_state = self.lstm(inputs)
        # pass lstm_output through additional dense layers for further processing if hidden_units were specified)
        if self.hidden_layers is not None:
            for l in self.hidden_layers:
                lstm_out = l(lstm_out)
        mix_params = self.mdn_out(lstm_out)
        return mix_params, h_state, c_state #, np.apply_along_axis(mdn.sample_from_output, 2, mix_params, self.output_dim, self.num_mixtures, temp=temperature) # uncomment to also compute samples from output params

    def train_op(self, inputs, targets, state):
        with tf.GradientTape() as tape:
            mix_params, _, _ = self(inputs, state)
            loss = self.loss_function(targets, mix_params)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
