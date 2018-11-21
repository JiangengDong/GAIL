import numpy as np
import gym
from numpy import (sqrt, arctan2, arccos, pi)
from typing import Tuple

from util import env_wrapper
from generator import Generator
from envs.reacher import reacher


class PIDPolicy:
    def __init__(self, shape: Tuple[int], p=0.2, i=0, d=2.4, ob_proc=None):
        self.p = p
        self.i = i
        self.d = d

        self.ob_proc = ob_proc

        self.sum = np.ndarray(shape)
        self.e_pre = np.ndarray(shape)

    def act(self, stochastic, ob):
        if self.ob_proc is not None:
            e = self.ob_proc(ob)
        else:
            e = ob
        inc = e - self.e_pre
        self.e_pre = e
        self.sum += e
        return (self.p * e + self.i * self.sum + self.d * inc), 0

    def __call__(self, stochastic, ob):
        return self.act(stochastic, ob)


def main():
    env = reacher(n_links=2)

    def ob_proc(ob):
        target = ob[:2]
        r = sqrt(target[0] ** 2 + target[1] ** 2)
        l1 = 0.1
        l2 = 0.11
        q_target = np.array([arctan2(target[1], target[0]) - arccos((r ** 2 + l1 ** 2 - l2 ** 2) / 2 / r / l1),
                             pi - arccos((l1 ** 2 + l2 ** 2 - r ** 2) / 2 / l1 / l2)])
        q = arctan2(ob[6:8], ob[4:6])
        return np.mod(q_target - q + pi, 2 * pi) - pi

    pol = PIDPolicy(shape=(2,), ob_proc=ob_proc)

    demo = Generator(pol, env, None, 1000, record_path="./record/demo.mp4")
    traj = demo.sample_trajectory(display=True, record=False)
    traj = demo.process_trajectory(traj, 0.995, 0.97)
    pass


if __name__ == '__main__':
    main()
