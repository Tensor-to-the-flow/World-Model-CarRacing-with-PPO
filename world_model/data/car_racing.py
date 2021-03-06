from gym.envs.box2d.car_racing import CarRacing
from gym.spaces.box import Box
from data.gatherer import process_frame


class CarRacingWrapper(CarRacing):

    def __init__(self, negative_range=True):
        super(CarRacingWrapper, self).__init__()
        # To deal with the processed images
        self.observation_space = Box(low=-1, high=1, shape=(1, 64, 64, 1))
        self.negative_range = negative_range

    def step(self, action):
        """ One step in the environment """
        frame, reward, done, info = super().step(action)

        # Solves bug where the image wasnt rendering
        self.viewer.window.dispatch_events()

        obs = process_frame(
            frame,
            vertical_cut=64,
            negative_range=self.negative_range
        )

        return obs, reward, done, info

    def reset(self):
        """ Resets the env and returns initial obs """
        raw = super().reset()
        # Solves bug where the image wasnt rendering
        self.viewer.window.dispatch_events()

        return raw
