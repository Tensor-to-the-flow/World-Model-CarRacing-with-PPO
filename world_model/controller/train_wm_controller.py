from world_model.controller.ppo_wm import Agent
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
    )[0]
    total_reward = 0
    data = defaultdict(list)
    obs = controller.env.reset()

    for step in range(episode_length):
        if render:
            controller.env.render("human")
        obs = obs.reshape(1, 64, 64, 1).astype(np.float32)
        # Collecting mu and logvar for data collection
        mu, logvar = vision.encode(obs)
        z = vision.reparameterize(mu, logvar)
        # encoded obs plus hidden state from memory --> controller input
        z_state = np.concatenate([z, state], axis=None)
        print(z_state.shape)
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
        y, h_state, c_state = memory(x, tf.cast(state, tf.float64), temperature=1.0)

        # Why is c_state part of the state that gets feed into the controller?
        #  Only plausible if c_state == latent space of
        state = [h_state, c_state]
        total_reward += reward

        if done:
            print('done at {} - reward {}'.format(step, reward))
            break

        if collect_data:
            reconstruct = vision.decoder(z)
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

    path = os.getcwd()[:-10]
    #memory = Memory(batch_size=1, num_timesteps=1)
    memory = Memory(num_timesteps=1, load_model=True, results_dir=path + 'memory/160model')
    env = CarRacingWrapper()
    #memory.load(path + 'memory/160model')
    vision = VAE()
    vision.load(path + 'vision')
    controller = Agent(env)

    r, d = train_controller(controller, vision, memory, collect_data=False)
    print(r)
