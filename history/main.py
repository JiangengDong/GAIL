import numpy as np

from GAIL import (Discriminator, Generator)
from generator import Demonstrator
from Demo_env import Demo_env
from Gene_env import Gene_env

from matplotlib import pyplot as plt


def discount_reward(reward):
    """discount reward by time: the longer the time, the less the influence

    :param np.ndarray reward: reward for each frame
    :return: reward given all the frame afterward
    """
    for i in range(len(reward)-2, -1, -1):
        reward[i] = reward[i] + reward[i+1]*0.99
    reward = reward/np.sum(reward)-np.mean(reward)
    return np.concatenate(([[0]], reward[0:50]))


def main():
    g = Generator(Gene_env)
    e = Demonstrator(Demo_env)
    d = Discriminator()

    for episode in range(1000):
        for i in range(10):
            l_expert = e.sample_locus()
            l_generator, _ = g.sample_locus()
            d.train(np.vstack((l_expert[1:], l_generator[1:])),
                    np.vstack((np.ones((50, 1)), np.zeros((50, 1)))))
        l_generator, a_generator = g.sample_locus()
        r_list = d.predict(l_generator)
        r_list = -np.log(1-r_list)
        r_list = discount_reward(r_list)
        g.train(l_generator[1:], a_generator[1:], r_list[1:])
        if episode % 10 == 0:
            print("episode %d" % (episode))
        if episode % 100 == 0:
            g.sample_locus(False, True)
            g.save()
            d.save()
    g.sample_locus(False, True)
    g.save()
    d.save()


if __name__ == '__main__':
    main()
