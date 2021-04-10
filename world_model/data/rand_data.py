from world_model.vision.vAE import VarAE
from world_model.data.car_racing import CarRacingWrapper
from world_model.data.gatherer import random_rollout
import os
import pickle
import tensorflow as tf
from numpy import savez_compressed
import numpy as np

N = 1000


def main(n):
    env = CarRacingWrapper()
    state = np.load('rr_data_state.npz')['arr_0']
    action = np.load('rr_data_action.npz')['arr_0']
    for i in range(n):
        bitch = random_rollout(env)
        state = np.concatenate((state, np.expand_dims(bitch['observation'], 0)))
        action = np.concatenate((action, np.expand_dims(bitch['action'], 0)))
        if i % 50 == 0 or i == n-1:
            print("Saving...")
            savez_compressed('rr_data_state.npz', state)
            savez_compressed('rr_data_action.npz', action)


def count():
    result = np.load('rr_data_state.npz')
    result = result['arr_0']
    print("Already saved datapoints: ", len(result))
    return len(result)


def drive():
    env = CarRacingWrapper()
    random_rollout(env)


def check():
    d = np.load('rr_data_action.npz')['arr_0']
    print(d.shape)
    print(d)
    print(len(d))


if __name__ == "__main__":
    #check()
    n = count()
    #main(N-n)
    #drive()
