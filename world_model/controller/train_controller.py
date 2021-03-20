import numpy as np
from .ppo import PPO
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

def train(model):

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    num_episodes = 20

    kwargs = {
        "model": PPO,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "continuous_normal_diagonal",
        "num_episodes": num_episodes,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_pg_lunarlander"

    buffer_size = 1000
    test_steps = 250
    epochs = 3
    sample_size = 100
    optim_batch_size = 24
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    print("Filling up the whole Experience Replay Buffer...")
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    #manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        print("Collecting data to train on...")
        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size, from_buffer=False)
        print(f"collected data for: {sample_dict.keys()}")
        #sample_dict['state'] = np.expand_dims(sample_dict['state'], -1)
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=None)
        print("optimizing...")

        # Collect the batched tf.datatsets
        state = data_dict['state']
        action = data_dict['action']
        reward = data_dict['reward']
        state_new = data_dict['state_new']
        not_done = data_dict['not_done']

        # Train loop lists
        losses = []
        future_reward = []
        r_storage = []
        log_probs = []

        for s, a, r, sn, nd in zip(state, action, reward, state_new, not_done):
            r_storage.append(r)
            # If 1 episode finished
            if nd == 0:
                current_future_reward = []
                acc_reward = 0
                # Compute future rewards for this episode
                for reward in reversed(r_storage):
                    acc_reward += reward
                    current_future_reward.append(acc_reward)
                # Collect future rewards for all episodes
                future_reward = tf.concat(future_reward, current_future_reward.reverse(), 0)
                r_storage = []
            # Compute log probabilities over all episodes
            log_probs = tf.concat(log_probs, agent.flowing_log_prob(s, a), 0)

        # Update weights
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(-(log_probs * future_reward)) / num_episodes
            gradients = tape.gradient(loss, agent.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

        losses.append(np.mean(loss))

        # set new weights
        manager.set_agent(agent.get_weights())
        # get new agent
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean(loss)}   avg env steps ::: {np.mean(time_steps)}"
        )

        """
        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)
        """

    # and load mmodels
    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True)


if __name__ == "__main__":
    model = PPO()
