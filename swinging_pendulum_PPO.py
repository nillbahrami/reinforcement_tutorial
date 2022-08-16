import tensorflow as tf


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PPO(object):

    def __init__(self):

        self.env = gym.make('Pendulum-v0').unwrapped
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_bound = [self.env.action_space.low, self.env.action_space.high]

        epsilon = 0.2

        self.sess = tf.Session()

        self.state_ph = tf.placeholder(tf.float32, [None, self.state_shape], 'state')

        with tf.variable_scope('value'):
            layer1 = tf.layers.dense(self.state_ph, 100, tf.nn.relu)
            self.v = tf.layers.dense(layer1, 1)

            self.Q = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

            self.advantage = self.Q - self.v

            self.value_loss = tf.reduce_mean(tf.square(self.advantage))

            self.train_value_nw = tf.train.AdamOptimizer(0.002).minimize(self.value_loss)

        pi, pi_params = self.build_policy_network('pi', trainable = True)

        oldpi, oldpi_params = self.build_policy_network('oldpi', trainable = False)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis = 0)

        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.action_ph = tf.placeholder(tf.float32, [None, self.action_shape], 'action')

        self.advantage_ph = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):

                ratio = pi.prob(self.action_ph) / oldpi.prob(self.action_ph)

                objective = ratio * self.advantage_ph

                L = tf.reduce_mean(tf.minimum(objective,
                                              tf.clip_by_value(ratio,
                                                               1. - epsilon,
                                                               1. + epsilon) * self.advantage_ph))

            self.policy_loss = -L

        with tf.variable_scope('train_policy'):
            self.train_policy_nw = tf.train.AdamOptimizer(0.001).minimize(self.policy_loss)

        self.sess.run(tf.global_variables_initializer())

    def train(self, state, action, reward):

        self.sess.run(self.update_oldpi_op)

        adv = self.sess.run(self.advantage, {self.state_ph: state, self.Q: reward})

        [self.sess.run(self.train_policy_nw,
                       {self.state_ph: state,
                        self.action_ph: action,
                        self.advantage_ph: adv}) for _ in range(10)]

        [self.sess.run(self.train_value_nw, {self.state_ph: state, self.Q: reward}) for _ in range(10)]


    def build_policy_network(self, name, trainable):
        with tf.variable_scope(name):

            layer1 = tf.layers.dense(self.state_ph, 100, tf.nn.relu, trainable = trainable)

            mu = 2 * tf.layers.dense(layer1, self.action_shape, tf.nn.tanh, trainable = trainable)

            sigma = tf.layers.dense(layer1, self.action_shape, tf.nn.softplus, trainable = trainable)

            norm_dist = tf.distributions.Normal(loc = mu, scale = sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = name)

        return norm_dist, params


    def select_action(self, state):

        state = state[np.newaxis, :]

        action = self.sess.run(self.sample_op, {self.state_ph: state})[0]

        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action


    def get_state_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.state_ph: state})[0, 0]


if __name__ == '__main__':

    gamma = 0.9

    tau = 0.01

    replay_buffer = 10000

    batch_size = 32

    num_episodes = 1000
    num_timesteps = 200

    ppo = PPO()

    for i in range(num_episodes):

        state = ppo.env.reset()

        episode_states, episode_actions, episode_rewards = [], [], []

        Return = 0

        for t in range(num_timesteps):

            ppo.env.render()

            action = ppo.select_action(state)

            next_state, reward, done, _ = ppo.env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append((reward + 8) / 8)

            state = next_state

            Return += reward

            if (t + 1) % batch_size == 0 or t == num_timesteps - 1:

                v_s_ = ppo.get_state_value(next_state)

                discounted_r = []
                for reward in episode_rewards[::-1]:
                    v_s_ = reward + gamma * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                es, ea, er = np.vstack(episode_states), np.vstack(episode_actions), np.array(discounted_r)[:, np.newaxis]

                episode_states, episode_actions, episode_rewards = [], [], []

                ppo.train(es, ea, er)

        if i % 10 == 0:
            print("Episode:{}, Return: {}".format(i, Return))

