import really
import gym
from collections import defaultdict
from PIL import Image
import numpy as np

RBG_WEIGHTS = [0.2989, 0.5870, 0.1140]


def random_rollout(env, episode_length=1000):
    """ runs an episode with a random policy """
    results = defaultdict(list)

    done = False
    observation = env.reset()
    step = 0
    while not done:
        #env.render("human")
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        transition = {'observation': np.expand_dims(np.expand_dims(observation, 0), -1), 'action': action}
        for key, data in transition.items():
            results[key].append(data)

        observation = next_observation
        step += 1
        if step >= episode_length:
            done = True

    env.close()

    return results


def process_frame(frame, screen_size=(64, 64), vertical_cut=4, max_val=255.0, negative_range=True):
    """crop, scale and converts the image to float"""
    frame = frame[:vertical_cut, :, :]
    frame = Image.fromarray(frame, mode='RGB')
    obs = frame.resize(screen_size, Image.BILINEAR)
    # RGB --> Greyscale
    obs = np.array(obs)
    obs = np.dot(obs[..., :3], RBG_WEIGHTS)
    # Normalize values
    #  Between 0-1
    obs = obs / max_val
    #  Between -1 and 1
    if negative_range:
        obs = obs * 2 - 1

    return obs
