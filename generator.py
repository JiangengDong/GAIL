import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Generator:
    def __init__(self, pi, env: gym.Env, reward_giver, n_step, record_path=None):
        self.pi = pi    # policy
        self.env = env  # environment for simulation
        self.reward_giver = reward_giver
        self.n_step = n_step
        self.path = record_path

        # Initialize history arrays
        self.obs = np.zeros((self.n_step,) + env.observation_space.shape, np.float32)
        self.acs = np.zeros((self.n_step,) + env.action_space.shape, np.float32)
        self.pre_acs = self.acs.copy()  # deep copy
        if self.reward_giver is not None:
            self.sts = np.ndarray((self.n_step,) + self.reward_giver.st_shape, np.float32)
        else:
            self.sts = None
        self.true_rewards = np.zeros(self.n_step, np.float32)
        self.rewards = np.zeros(self.n_step, np.float32)
        self.vpreds = np.zeros(self.n_step, np.float32)
        self.news = np.zeros(self.n_step, np.int32)
        pass

    def sample_trajectory(self, stochastic=True, display=False, record=False):
        # Initialize state variables
        t = 0
        ac = self.env.action_space.sample()
        new = True  # whether a new episode begins
        reward = 0.0  # reward predicted by value function
        true_reward = 0.0  # reward calculated according to all rewards
        vpred = 0.0
        if self.reward_giver is not None:
            st = np.zeros(self.reward_giver.st_shape, np.float32)
        ob = self.env.reset()

        if record:
            rec = VideoRecorder(self.env, path=self.path)

        for i in range(self.n_step):
            # record the previous data
            pre_ac = ac
            self.obs[i] = ob
            self.pre_acs[i] = pre_ac
            self.news[i] = new
            # perform policy and record
            ac, vpred = self.pi.act(stochastic, ob)
            self.acs[i] = ac
            self.vpreds[i] = vpred
            # evaluate values and record
            if self.reward_giver is not None:
                reward, st = self.reward_giver.get_reward(ob, st)
                self.sts[i] = st
            else:
                reward = 0
            self.rewards[i] = reward
            # take action and record true reward
            ob, true_reward, new, _ = self.env.step(ac)
            if record:
                rec.capture_frame()
            elif display:
                self.env.render()
            self.true_rewards[i] = true_reward
            if new:
                ob = self.env.reset()
                if self.reward_giver is not None:
                    st = np.zeros(self.reward_giver.st_shape, np.float32)

        if display:
            self.env.close()
        return {"ob": self.obs,
                "reward": self.rewards,
                "vpred": self.vpreds,
                "ac": self.acs,
                "pre_ac": self.pre_acs,
                "new": self.news,
                "nextvpred": vpred * (1 - new),
                "st": self.sts}

    @staticmethod
    def process_trajectory(traj, gamma, lam):
        new = np.append(traj["new"],
                        0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(traj["vpred"], traj["nextvpred"])
        T = len(traj["reward"])
        traj["adv"] = gaelam = np.empty(T, 'float32')
        rewards = traj["reward"]
        # discounted rewards:
        # A_pre + v_pre = gamma*(A*lambda + v) + r_pre
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        traj["tdlamret"] = traj["adv"] + traj["vpred"]  # target of value function
        return traj
