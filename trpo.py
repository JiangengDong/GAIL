import tensorflow as tf
import numpy as np
import gym
from typing import List, Optional

from multiLayerPolicy import MultiLayerPolicy
from discriminator import Discriminator
from generator import Generator


def cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """Conjunct gradient method."""
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


class TRPO:
    def __init__(self, env,
                 ent_coeff=0,
                 max_episode=100000,
                 g_step=3,
                 d_step=1,
                 vf_step=3,
                 gamma=0.995,
                 lam=0.97,
                 max_kl=0.01,
                 cg_damping=0.1):
        self.env = env
        self.ent_coeff = ent_coeff
        self.max_episode = max_episode
        self.g_step = g_step
        self.d_step = d_step
        self.vf_step = vf_step
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.build_net(env, ent_coeff)

    def build_net(self, env, ent_coeff):
        # Build two policies. Optimize the performance of pi w.r.t oldpi in each step.
        self.ob = tf.placeholder(dtype=tf.float32, shape=(None,) + env.observation_space.shape, name="ob")
        self.pi = MultiLayerPolicy("pi", self.ob, env.action_space.shape)
        self.oldpi = MultiLayerPolicy("pi", self.ob, env.action_space.shape)
        self.assignNewToOld = [tf.assign(oldv, newv) for (oldv, newv) in
                               zip(self.oldpi.get_variables(), self.pi.get_variables())]
        # Build discriminator.
        self.d = Discriminator(name="discriminator",
                               ob_shape=env.observation_space.shape,
                               ac_shape=env.action_space.shape)
        # Build generator.
        self.g = Generator(self.pi, env, self.d, 50)

        # KL divergence and entropy
        self.meanKl = tf.reduce_mean(self.oldpi.pd.kl(self.pi.pd))  # D_KL using Monte Carlo on a batch
        meanEnt = tf.reduce_mean(self.pi.pd.entropy())  # entropy using Monte Carlo on a batch
        entBonus = ent_coeff * meanEnt
        # surrogate gain, L(pi)=J(pi)-J(oldpi)=sum(p_new/p_old*adv)
        self.ac = tf.placeholder(dtype=tf.float32, shape=(None,) + env.action_space.shape, name="ac")
        self.atarg = tf.placeholder(dtype=tf.float32, shape=(None,),
                                    name="advantage")  # advantage function for each action
        ratio = tf.exp(self.pi.pd.logp(self.ac) - self.oldpi.pd.logp(self.ac))  # p_new/p_old
        surrgain = tf.reduce_mean(ratio * self.atarg)  # J(pi)-J(oldpi)
        self.optimgain = surrgain + entBonus
        # fisher vector product
        all_var_list = self.pi.get_trainable_variables()
        policyVars = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
        self.vector = tf.placeholder(dtype=tf.float32, shape=(None,), name="vector")
        self.fvp = self.build_fisher_vector_product(self.meanKl, self.vector, policyVars)
        # loss and gradient
        self.optimgrad = tf.gradients(self.optimgain, policyVars)
        # utils
        self.init = tf.global_variables_initializer()
        self.get_theta = tf.concat([tf.reshape(var, [int(np.prod(var.shape))]) for var in policyVars], axis=0)
        self.set_theta = self.setFromTheta(policyVars)

    def setFromTheta(self, var_list):
        # count the number of elements in var_list
        shape_list = [var.shape.as_list() for var in var_list]
        size_list = [int(np.prod(shape)) for shape in shape_list]
        total_size = np.sum(size_list)
        self.theta = tf.placeholder(dtype=tf.float32, shape=(total_size,), name="theta")
        # assign var_list from the given theta
        assign_list = []
        start = 0
        for (shape, var, size) in zip(shape_list, var_list, size_list):
            assign_list.append(tf.assign(var, tf.reshape(self.theta[start, start+size], shape)))
            start += size
        return tf.group(*assign_list)

    def build_fisher_vector_product(self, kl, vector, var_list):
        kl_grad_list = tf.gradients(kl, var_list)
        # transform vector into the same shape of matrix in var_list
        shape_list = [var.shape.as_list() for var in var_list]
        start = 0
        vector_list = []
        n_list = []
        for shape in shape_list:
            n = int(np.prod(shape))
            vector_list.append(tf.reshape(vector[start:start + n], shape))
            n_list.append(n)
            start += n
        # element-wise product of kl_grad and vector, then add all the elements together
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(kl_grad_list, vector_list)])
        fvp_list = tf.gradients(gvp, var_list)
        fvp = tf.concat([tf.reshape(fvp_part, (n,)) for (fvp_part, n) in zip(fvp_list, n_list)])
        return fvp

    def train(self):
        with tf.Session() as sess:
            # Preparation
            fvp = lambda p, ob, ac, adv: sess.run(self.fvp,
                                                  {self.ob: ob, self.ac: ac,
                                                   self.atarg: adv, self.vector: p}) + self.cg_damping * p
            set_theta = lambda theta: sess.run(self.set_theta, {self.theta: theta})
            get_theta = lambda: sess.run(self.get_theta)
            get_kl = lambda ob, ac, adv: sess.run(self.meanKl,
                                                  {self.ob: ob, self.ac: ac, self.atarg: adv})

            # Start training
            sess.run(self.init)
            for episode in range(self.max_episode):
                for _ in range(self.g_step):
                    # sample trajectory
                    traj = self.g.sample_trajectory()
                    traj = self.g.process_trajectory(traj, self.gamma, self.lam)
                    obs, acs, advs, vtarg, vpred = traj["ob"], traj["ac"], traj["adv"], traj["tdlamret"], traj["vpred"]
                    # normalization
                    advs = (advs - advs.mean()) / advs.std()  # advantage is normalized on a batch
                    self.pi.ob_rms.update(obs)  # observation is normalized on all history data
                    # data required to calculate fisher vector product
                    fvpargs = obs, acs, advs

                    # loss and gradients on this batch
                    loss, g = sess.run([self.optimgain, self.optimgrad],
                                       {self.ob: obs, self.ac: acs, self.atarg: advs})

                    if not np.allclose(g, 0):
                        # use conjunct gradient method to solve Hs=g, where H = nabla^2 D_KL
                        stepdir = cg(lambda p: fvp(p, obs, acs, advs), g)
                        sHs = 0.5*stepdir.dot(fvp(stepdir, obs, acs, advs))
                        lm = np.sqrt(sHs/self.max_kl)
                        fullstep = stepdir/lm   # get step from direction
                        expertedimprove = g.dot(fullstep)
                        surrogate_gain_before = loss
                        stepsize = 1.0
                        theta_before = get_theta()
                        for _ in range(10):
                            theta_new = theta_before + fullstep*stepsize
                            set_theta(theta_new)
                            surr, kl = sess.run([self.optimgain, self.meanKl],
                                                {self.ob: obs, self.ac: acs, self.atarg: advs})
                            if kl > 1.5*self.max_kl:
                                pass    # violate kl constraint
                            elif surr-surrogate_gain_before < 0:
                                pass    # surrogate gain not improve
                            else:
                                break   # stepsize OK
                            stepsize *= 5
                        else:
                            set_theta(theta_before)     # find no good step
                    for _ in range(self.vf_step):
                        # TODO: expert trajectory, the same as generator
                        self.pi.ob_rms.update(obs)
                        self.pi.train_value_function(obs, vtarg)
