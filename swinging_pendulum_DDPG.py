import tensorflow as tf


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DDPG(object):

    def __init__(self, state_shape, action_shape, high_action_value, ):

        self.replay_buffer = np.zeros((replay_buffer,
                                       state_shape * 2 + action_shape + 1), dtype = np.float32)

        self.num_transitions = 0

        self.sess = tf.Session()

        self.noise = 3.0

        self.state_shape, self.action_shape, self.high_action_value = state_shape, action_shape, high_action_value
        self.state = tf.placeholder(tf.float32, [None, state_shape], 'state')
        self.next_state = tf.placeholder(tf.float32, [None, state_shape], 'next_state')

        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        with tf.variable_scope('Actor'):
            self.actor = self.build_actor_network(self.state, scope = 'main', trainable = True)
            target_actor = self.build_actor_network(self.next_state, scope = 'target', trainable = False)

        with tf.variable_scope('Critic'):
            critic = self.build_critic_network(self.state, self.actor, scope = 'main', trainable = True)
            target_critic = self.build_critic_network(self.next_state, target_actor, scope = 'target', trainable = False)

        self.main_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/main')
        self.target_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/target')

        self.main_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/main')
        self.target_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/target')

        self.soft_replacement = [
            [tf.assign(phi_, tau * phi + (1 - tau) * phi_), tf.assign(theta_, tau * theta + (1 - tau) * theta_)]
            for phi, phi_, theta, theta_ in
            zip(self.main_actor_params, self.target_actor_params, self.main_critic_params, self.target_critic_params)
        ]

        y = self.reward + gamma * target_critic

        MSE = tf.losses.mean_squared_error(labels = y, predictions = critic)

        self.train_critic = tf.train.AdamOptimizer(0.01).minimize(MSE, name = "adam-ink",
                                                                  var_list = self.main_critic_params)

        actor_loss = -tf.reduce_mean(critic)

        self.train_actor = tf.train.AdamOptimizer(0.001).minimize(actor_loss, var_list = self.main_actor_params)

        self.sess.run(tf.global_variables_initializer())


    def select_action(self, state):
        action = self.sess.run(self.actor, {self.state: state[np.newaxis, :]})[0]

        action = np.random.normal(action, self.noise)

        action = np.clip(action, action_bound[0], action_bound[1])

        return action


    def train(self):

        self.sess.run(self.soft_replacement)

        indices = np.random.choice(replay_buffer, size = batch_size)

        batch_transition = self.replay_buffer[indices, :]

        batch_states = batch_transition[:, :self.state_shape]
        batch_actions = batch_transition[:, self.state_shape: self.state_shape + self.action_shape]
        batch_rewards = batch_transition[:, -self.state_shape - 1: -self.state_shape]
        batch_next_state = batch_transition[:, -self.state_shape:]

        self.sess.run(self.train_actor, {self.state: batch_states})

        self.sess.run(self.train_critic, {self.state: batch_states, self.actor: batch_actions,
                                          self.reward: batch_rewards, self.next_state: batch_next_state})

    def store_transition(self, state, actor, reward, next_state):

        trans = np.hstack((state, actor, [reward], next_state))

        index = self.num_transitions % replay_buffer

        self.replay_buffer[index, :] = trans

        self.num_transitions += 1

        if self.num_transitions > replay_buffer:
            self.noise *= 0.99995
            self.train()

    def build_actor_network(self, state, scope, trainable):

        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(state, 30,
                                      activation = tf.nn.tanh,
                                      name = 'layer_1',
                                      trainable = trainable)
            actor = tf.layers.dense(layer_1,
                                    self.action_shape,
                                    activation = tf.nn.tanh,
                                    name = 'actor',
                                    trainable = trainable)

            return tf.multiply(actor, self.high_action_value, name = "scaled_a")

    def build_critic_network(self, state, actor, scope, trainable):

        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.state_shape, 30], trainable = trainable)
            w1_a = tf.get_variable('w1_a', [self.action_shape, 30], trainable = trainable)
            b1 = tf.get_variable('b1', [1, 30], trainable = trainable)
            net = tf.nn.tanh(tf.matmul(state, w1_s) + tf.matmul(actor, w1_a) + b1)

            critic = tf.layers.dense(net, 1, trainable = trainable)

            return critic


if __name__ == '__main__':

    env = gym.make("Pendulum-v0").unwrapped

    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]

    gamma = 0.9

    tau = 0.01

    replay_buffer = 10000

    batch_size = 32

    num_episodes = 300
    num_timesteps = 500

    ddpg = DDPG(state_shape, action_shape, action_bound[1])

    for i in range(num_episodes):

        state = env.reset()

        Return = 0

        for j in range(num_timesteps):

            env.render()

            action = ddpg.select_action(state)

            next_state, reward, done, info = env.step(action)

            ddpg.store_transition(state, action, reward, next_state)

            Return += reward

            if done:
                break

            state = next_state

        if i % 10 == 0:
            print("Episode:{}, Return: {}".format(i, Return))


