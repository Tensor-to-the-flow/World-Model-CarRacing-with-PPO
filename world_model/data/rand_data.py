from world_model.vision.vAE import VarAE
from world_model.data.car_racing import CarRacingWrapper
from world_model.data.gatherer import random_rollout
import os
import pickle
import tensorflow as tf
from numpy import savez_compressed
import numpy as np

N = 10000


def main(n):
    env = CarRacingWrapper()
    result = np.load('rr_data.npz')['arr_0']
    for i in range(n):
        result = np.concatenate((result, np.expand_dims(random_rollout(env)['observation'], 0)))
        if i % 50 == 0 or i == n-1:
            print("Saving")
            savez_compressed('rr_data.npz', result)


def count():
    result = np.load('rr_data.npz')
    result = result['arr_0']
    print("Already saved datapoints: ", len(result))
    return len(result)


if __name__ == "__main__":
    n = count()
    #main(N-n)
