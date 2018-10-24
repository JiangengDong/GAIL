import numpy as np
from numpy import (arctan2, arccos, sqrt, pi)
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from Demo_env import Demo_env


class PID:
    def __init__(self, p=0.2, i=0, d=2.4):
        self.p = p
        self.i = i
        self.d = d

        self.sum = 0
        self.e_pre = 0

    def u(self, e):
        inc = e - self.e_pre
        self.e_pre = e
        self.sum += e
        return self.p * e + self.i * self.sum + self.d * inc

    def __call__(self, *args, **kwargs):
        return self.u(*args, **kwargs)


class Demonstrator:
    def __init__(self, env):
        self.pid0 = PID()
        self.pid1 = PID()
        self.env = env
        self.episode = 0
        # rec = VideoRecorder(env, path='./record/test.mp4')

    def sample_locus(self, record=False, show=False):
        self.episode += 1
        # find a reachable target
        while True:
            observation = self.env.reset()
            target = self.env.unwrapped.goal
            r = sqrt(target[0] ** 2 + target[1] ** 2)
            l1 = 0.1
            l2 = 0.11
            if abs(l1-l2) < r < l1+l2:
                break

        # calculate angle for each joint
        q_target = np.array([arctan2(target[1], target[0]) - arccos((r ** 2 + l1 ** 2 - l2 ** 2) / 2 / r / l1),
                             pi - arccos((l1 ** 2 + l2 ** 2 - r ** 2) / 2 / l1 / l2)])

        x_pre = observation[8:10]
        x_now = observation[8:10]
        v_pre = np.array([0, 0])
        v_now = np.array([0, 0])
        a_now = np.array([0, 0])
        state_context_list = np.hstack((x_now, v_now, a_now, target))
        action_list = np.array([None, None])
        action = [0, 0]
        for _ in range(50):
            # rec.capture_frame()
            if show:
                self.env.render()
            observation, _, _, _ = self.env.step(action)
            x_now = observation[8:10]
            v_now = (x_now-x_pre)*50
            a_now = (v_now-v_pre)*50
            x_pre = x_now
            v_pre = v_now
            state_context_list = np.vstack((state_context_list, np.hstack((x_now, v_now, a_now, target))))
            q = arctan2(observation[2:4], observation[0:2])
            e = np.mod(q_target - q + pi, 2 * pi) - pi
            action = [self.pid0(e[0]), self.pid1(e[1])]
            action_list = np.vstack((action_list, action))

        if record:
            np.savez("./record/locus/Demo_%d.npz"%self.episode, target, state_context_list)
        return state_context_list
        # rec.close()


if __name__ == '__main__':
    a = Demonstrator(Demo_env)
    for _ in range(30):
        L = a.sample_locus(False, True)
        pass
    # env = gym.make('Reacher-v2')
    # env.render = MethodType(render, env.env)
    # rec = VideoRecorder(env, path='./record/test.mp4')
    # for _ in range(30):
    #     env.reset()
    #     for _ in range(50):
    #         rec.capture_frame()
    #         env.step(env.action_space.sample())
    # rec.close()