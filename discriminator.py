import tensorflow as tf
import numpy as np
from util import RunningMeanStd


def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits + tf.nn.softplus(-logits)
    return ent


class Discriminator:
    def __init__(self, name, ob_shape, ac_shape, hid_size=128, ent_coff=0.001):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self.build_net(ob_shape, ac_shape, hid_size, ent_coff)
            self.writer = tf.summary.FileWriter("./log/discriminator")

    def build_net(self, ob_shape, ac_shape, hid_size, ent_coeff):
        # build placeholders
        self.generator_obs = tf.placeholder(tf.float32, (None,) + ob_shape, name="observations")
        self.generator_acs = tf.placeholder(tf.float32, (None,) + ac_shape, name="actions")
        self.expert_obs = tf.placeholder(tf.float32, (None,) + ob_shape, name="expert_observations")
        self.expert_acs = tf.placeholder(tf.float32, (None,) + ac_shape, name="expert_actions")

        # normalize observation
        with tf.variable_scope("obfilter"):
            self.obs_rms = RunningMeanStd(shape=ob_shape)

        # network to judge generator
        net = (self.generator_obs-self.obs_rms.mean)/self.obs_rms.std
        net = tf.concat([net, self.generator_acs], axis=1)
        with tf.variable_scope("main_net", reuse=False):
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            generator_logits = tf.layers.dense(inputs=net, units=1, activation=tf.identity)
        # network to judge expert
        net = (self.expert_obs-self.obs_rms.mean)/self.obs_rms.std
        net = tf.concat([net, self.expert_acs], axis=1)
        with tf.variable_scope("main_net", reuse=True):
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            expert_logits = tf.layers.dense(inputs=net, units=1, activation=tf.identity)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        self.merged = tf.summary.merge([tf.summary.scalar("Expert accuracy", expert_acc),
                                        tf.summary.scalar("Generator accuracy", generator_acc)])

        # loss for the two networks respectively
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -ent_coeff * entropy

        # reward and optimizer
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.adam = tf.train.AdamOptimizer().minimize(loss=self.total_loss,
                                                      var_list=self.get_trainable_variable())

    def get_trainable_variable(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs: obs, self.generator_acs: acs}
        return tf.get_default_session().run(self.reward_op, feed_dict)

    def train(self, generator_obs, generator_acs, expert_obs, expert_acs):
        _, summary = tf.get_default_session().run([self.adam, self.merged],
                                                  {self.generator_obs: generator_obs,
                                                   self.generator_acs: generator_acs,
                                                   self.expert_obs: expert_obs,
                                                   self.expert_acs: expert_acs})
        try:
            self.summary_step += 1
        except AttributeError:
            self.summary_step = 0
        finally:
            self.writer.add_summary(summary, self.summary_step)


def main():
    d = Discriminator("discriminator", (11,), (2,), 1024)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        pass


if __name__ == '__main__':
    main()
