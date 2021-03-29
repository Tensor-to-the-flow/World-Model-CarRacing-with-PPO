from world_model.vision.vAE import VarAE
from world_model.data.car_racing import CarRacingWrapper
from world_model.data.gatherer import random_rollout
import os
import pickle
import numpy as np
import tensorflow as tf

N = 1


def main():
    #v = VarAE()
    #env = CarRacingWrapper()
    path = os.getcwd()
    path = path[:-6] + 'data/rr_data.npz'
    rand_data = np.load(path)['arr_0']
    print()

    ds = tf.data.Dataset.from_tensor_slices(rand_data)



if __name__ == "__main__":
    main()
