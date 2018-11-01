import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import (sqrt, pi, exp, square, matmul, prod)
import time
import gym

import tensorflow as tf
from tensorflow.contrib import (slim, layers)

from Gene_env import Gene_env


class Generator:
    def __init__(self, env):
        # tensorflow Graph and Session
        self.graph = tf.Graph()
        self.__build_net()
        self.writer = tf.summary.FileWriter("./log/generator", self.graph)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        # OpenAI Gym
        self.env = env  # type: gym.wrappers.TimeLimit
        self.episode = 0  # type: int

    def __del__(self):
        self.sess.close()
        self.writer.close()
        print('Generator session is closed.')

    def __build_net(self):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.tf_s = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='state')
                self.tf_a = tf.placeholder(dtype=tf.float32, shape=[None, 2, 1], name='action')
                self.tf_r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')

            with tf.name_scope('layers'):
                net = layers.fully_connected(self.tf_s, 1024,
                                             activation_fn=tf.nn.sigmoid,
                                             normalizer_fn=layers.batch_norm,
                                             weights_regularizer=layers.l2_regularizer(2.5e-5))
                self.mean = layers.fully_connected(net, 2,
                                                   activation_fn=tf.nn.tanh,
                                                   normalizer_fn=None,
                                                   weights_regularizer=layers.l2_regularizer(2.5e-5))
                self.sqrt_var = layers.fully_connected(net, 4,
                                                       activation_fn=tf.nn.tanh,
                                                       normalizer_fn=None,
                                                       weights_regularizer=layers.l2_regularizer(2.5e-5))

            with tf.name_scope('output'):
                self.mean = tf.reshape(self.mean, [-1, 2, 1])
                self.sqrt_var = tf.reshape(tf.multiply(self.sqrt_var, tf.constant([1, 0, 1, 1], tf.float32)),
                                           [-1, 2, 2])

            with tf.name_scope('loss'):
                prob = tf.matmul(tf.matrix_inverse(self.sqrt_var), (self.tf_a - self.mean))
                prob = tf.exp(prob)
                self.loss = tf.reduce_sum(tf.reduce_prod(prob, 1)*self.tf_r)
                tf.summary.scalar("Generator_loss", self.loss)

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss*(-1))

            with tf.name_scope('util'):
                self.saver = tf.train.Saver()
                self.init = tf.global_variables_initializer()
                self.summary = tf.summary.merge_all()

    def sample_locus(self, record=False, show=False):
        # find a reachable target
        while True:
            observation = self.env.reset()
            target = self.env.unwrapped.goal
            r = sqrt(target[0] ** 2 + target[1] ** 2)
            l1 = 0.1
            l2 = 0.11
            if abs(l1 - l2) < r < l1 + l2:
                break

        x_pre = observation[8:10]
        x_now = observation[8:10]
        v_pre = np.array([0, 0])
        v_now = np.array([0, 0])
        a_now = np.array([0, 0])
        state_context_list = np.hstack((x_now, v_now, a_now, target)).reshape((1, -1))
        action_list = np.array([[[0], [0]]])
        action = np.array([0, 0])
        for _ in range(50):
            # rec.capture_frame()
            if show:
                self.env.render()
            # s_{t+1}~P(s_{t+1}|a_t, s_t)
            observation, _, _, _ = self.env.step(action.flatten())
            x_now = observation[8:10]
            v_now = (x_now - x_pre) * 50
            a_now = (v_now - v_pre) * 50
            x_pre = x_now
            v_pre = v_now
            state_context_list = np.vstack((state_context_list, np.hstack((x_now, v_now, a_now, target))))
            # sample action
            mean, sqrt_var = self.sess.run([self.mean, self.sqrt_var], {self.tf_s: state_context_list[-1:][:]})
            z = np.random.standard_normal((2, 1))
            action = (matmul(sqrt_var, z) + mean).T     # type: np.ndarray
            if action_list is None:
                action_list = np.reshape(action, (1, 2, 1))
            else:
                action_list = np.vstack((action_list, action))

        if record:
            np.savez("./record/locus/Gene_%d.npz" % self.episode, target, state_context_list)
        self.episode += 1
        return state_context_list, action_list
        # rec.close()

    def train(self, state_context_list, action_list, reward_list):
        _, summary = self.sess.run([self.train_op, self.summary],
                                {self.tf_a: action_list, self.tf_r: reward_list, self.tf_s: state_context_list})
        self.writer.add_summary(summary, self.episode)

    def save(self):
        self.saver.save(self.sess, "./log/generator/generator.ckpt")

    def load(self):
        self.saver.restore(self.sess, "./log/generator/generator.ckpt")


class Discriminator:
    def __init__(self):
        self.graph = tf.Graph()
        self.__build_net()
        self.writer = tf.summary.FileWriter("./log/discriminator", self.graph)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.episode = 0

    def __del__(self):
        self.sess.close()
        self.writer.close()
        print('Discriminator session is closed.')

    def __build_net(self):
        with self.graph.as_default():
            with tf.name_scope("input"):
                self.tf_s = tf.placeholder(dtype=tf.float32, shape=(None, 8), name="state_context")
                self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="label")

            with tf.name_scope("layers"):
                net = layers.fully_connected(self.tf_s, 1024,
                                             activation_fn=tf.nn.tanh,
                                             normalizer_fn=layers.batch_norm,
                                             weights_regularizer=layers.l2_regularizer(2.5e-5))
                logits = layers.fully_connected(net, 1,
                                                activation_fn=None,
                                                normalizer_fn=layers.batch_norm,
                                                weights_regularizer=layers.l2_regularizer(2.5e-5))

            with tf.name_scope("output"):
                self.prob = tf.nn.sigmoid(logits)

            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tf_label, logits=logits))
                tf.summary.scalar("Discriminator_loss", self.loss)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

            with tf.name_scope('util'):
                self.saver = tf.train.Saver()
                self.init = tf.global_variables_initializer()
                self.summary = tf.summary.merge_all()

    def train(self, state_context_list, prob_true):
        _, summary = self.sess.run([self.train_op, self.summary],
                                {self.tf_label: prob_true, self.tf_s: state_context_list})
        self.writer.add_summary(summary, self.episode)
        self.episode += 1

    def predict(self, state):
        return self.sess.run(self.prob, {self.tf_s: state})

    def save(self):
        self.saver.save(self.sess, "./log/discriminator/discriminator.ckpt")

    def load(self):
        self.saver.restore(self.sess, "./log/discriminator/discriminator.ckpt")


if __name__ == '__main__':
    g = Generator(Gene_env)
    d = Discriminator()
    g.load()
    d.load()
    for _ in range(30):
        s, a = g.sample_locus(False, True)
    del g
    del d
    pass
