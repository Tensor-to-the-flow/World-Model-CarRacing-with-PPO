from gym.envs.box2d.car_racing import CarRacing
from gym.spaces.box import Box
from world_model.data.gatherer import process_frame


class CarRacingWrapper(CarRacing):

    def __init__(self):
        super(CarRacingWrapper, self).__init__()
        # To deal with the processed images
        self.observation_space = Box(low=-1, high=1, shape=(1, 64, 64, 1))

    def step(self, action):
        """ One step in the environment """
        frame, reward, done, info = super().step(action)

        # Solves bug where the image wasnt rendering
        self.viewer.window.dispatch_events()

        obs = process_frame(
            frame,
            vertical_cut=64
        )

        return obs, reward, done, info

    def reset(self):
        """ Resets the env and returns initial obs """
        raw = super().reset()
        # Solves bug where the image wasnt rendering
        self.viewer.window.dispatch_events()

        return raw
