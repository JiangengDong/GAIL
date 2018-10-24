import gym
import numpy as np
from types import MethodType
import mpi4py

def render(self, mode='human'):
    if mode == 'rgb_array':
        self._get_viewer().render()
        # window size used for old mujoco-py:
        width, height = 1920, 1080
        data = self._get_viewer().read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'human':
        self._get_viewer().render()


def _get_obs(self):
    theta = self.sim.data.qpos.flat[:2]
    return np.concatenate([
        np.cos(theta),
        np.sin(theta),
        self.sim.data.qpos.flat[2:],
        self.sim.data.qvel.flat[:2],
        self.get_body_com("fingertip")
    ])


Demo_env = gym.make('Reacher-v2')
Demo_env.env.render = MethodType(render, Demo_env.env)
Demo_env.env._get_obs = MethodType(_get_obs, Demo_env.env)
pass