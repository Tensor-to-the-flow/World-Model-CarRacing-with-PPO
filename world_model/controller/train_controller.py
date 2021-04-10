from world_model.controller.ppo import Agent
from world_model.data.car_racing import CarRacingWrapper
import numpy as np
import tensorflow as tf
from collections import defaultdict
from world_model.vision.vAE import VAE
from world_model.memory.memory import Memory
import os


def train_controller(controller, vision, memory, collect_data=True, episode_length=1000, render=False):
    """ runs a single episode """
    #  needs to be imported here for multiprocessing

    state = memory.lstm.get_zero_hidden_state(
        np.zeros(35).reshape(1, 1, 35)
    )

    total_reward = 0
    data = defaultdict(list)
    obs = controller.env.reset()
    print("STATE : ", state)
    for step in range(episode_length):
        if render:
            controller.env.render("human")
        obs = obs.reshape(1, 64, 64, 1).astype(np.float32)
        # Collecting mu and logvar for data collection
        mu, logvar = vision.encode(obs)
        z = vision.reparameterize(mu, logvar)
        # encoded obs plus hidden state from memory --> controller input
        z_state = np.concatenate([z, state], axis=None)

        print("Z-state: \n", z_state)
        #action = controller.actor.get_action(z_state)
        action = controller.env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        action = tf.cast(action, tf.float64)
        x = tf.concat([
            tf.reshape(z, (1, 1, 32)),
            tf.reshape(action, (1, 1, 3))
        ], axis=2)
        # state :: hidden_state (,256) & cell_state (,256)
        # Shouldnt it be hidden_state and prediction of new state space z? (Mixture-ouput y)?
        # NOOO we dont use the pred_z (y) for training the controller
        print("X: ", x)
        y, h_state, c_state = memory(x, tf.cast(state, tf.float64), temperature=1.0)

        # Why is c_state part of the state that gets feed into the controller?
        #  Only plausible if c_state == latent space of
        state = [h_state, c_state]
        total_reward += reward
        print("STATE NEW: \n", state)

        if done:
            print('done at {} - reward {}'.format(step, reward))
            break

        if collect_data:
            reconstruct = vision.decode(z)
            vae_loss = vision.loss(reconstruct)
            data['observation'].append(obs)
            data['latent'].append(np.squeeze(z))
            data['reconstruct'].append(np.squeeze(reconstruct))
            data['reconstruction-loss'].append(vae_loss['reconstruction-loss']),
            data['unclipped-kl-loss'].append(vae_loss['unclipped-kl-loss'])
            data['action'].append(action)
            data['mu'].append(mu)
            data['logvar'].append(logvar)
            data['pred-latent'].append(y)
            data['pred-reconstruct'].append(np.squeeze(vision.decode(y.reshape(1, 32))))
            data['total-reward'].append(total_reward)

    env.close()
    return total_reward, data


if __name__ == "__main__":

    memory = Memory(batch_size=1, num_timesteps=1)
    env = CarRacingWrapper()
    path = os.getcwd()[:-10]
    memory.load(path + 'memory/160model')
    vision = VAE()
    vision.load(path + 'vision')
    controller = Agent(env)

    train_controller(controller, vision, memory)
