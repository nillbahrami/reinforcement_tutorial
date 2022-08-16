import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
from collections import deque
import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import gym
from tensorflow.python.framework import ops


class Categorical_DQN():

    # first, let's define the init method
    def __init__(self, env,
                 v_min = 0,
                 v_max = 1000,
                 atoms = 51,
                 gamma = 0.99,
                 batch_size = 10,
                 update_target_net = 50,
                 epsilon = 0.5,
                 buffer_length = 20000
                 ):

        self.v_min = v_min
        self.v_max = v_max

        self.atoms = atoms

        self.gamma = gamma

        self.batch_size = batch_size

        self.update_target_net = update_target_net

        self.epsilon = epsilon

        self.buffer_length = buffer_length
        self.replay_buffer = deque(maxlen = self.buffer_length)

        # start the TensorFlow session
        self.sess = tf.InteractiveSession()

        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

        self.time_step = 0

        # initialize the target state shape
        target_state_shape = [1]
        target_state_shape.extend(self.state_shape)

        # define the placeholder for the state
        self.state_ph = tf.placeholder(tf.float32, target_state_shape)

        # define the placeholder for the action
        self.action_ph = tf.placeholder(tf.int32, [1, 1])

        # define the placeholder for the m value (distributed probability of target distribution)
        self.m_ph = tf.placeholder(tf.float32, [self.atoms])

        # compute delta z
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)

        # compute the support values
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]

        self.build_categorical_DQN()

        # initialize all the TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

    # define a function called build_network for building a deep network. Since we are
    # dealing with the Atari games, we use the convolutional neural network
    def build_network(self, state,
                      action,
                      name,
                      units_1,
                      units_2,
                      weights,
                      bias,
                      reg = None):

        # define the first convolutional layer
        with tf.variable_scope('conv1'):
            conv1 = self.conv(state, [5, 5, 3, 6], [6], [1, 2, 2, 1], weights, bias)

        # define the second convolutional layer
        with tf.variable_scope('conv2'):
            conv2 = self.conv(conv1, [3, 3, 6, 12], [12], [1, 2, 2, 1], weights, bias)

        # flatten the feature maps obtained as a result of the second convolutional layer
        with tf.variable_scope('flatten'):
            flatten = tf.layers.flatten(conv2)

        # define the first dense layer
        with tf.variable_scope('dense1'):
            dense1 = self.dense(flatten, units_1, [units_1], weights, bias)

        # define the second dense layer
        with tf.variable_scope('dense2'):
            dense2 = self.dense(dense1, units_2, [units_2], weights, bias)

        # concatenate the second dense layer with the action
        with tf.variable_scope('concat'):
            concatenated = tf.concat([dense2, tf.cast(action, tf.float32)], 1)

        # define the third layer and apply the softmax function to the result of the third layer and
        # obtain the probabilities for each of the atoms
        with tf.variable_scope('dense3'):
            dense3 = self.dense(concatenated, self.atoms, [self.atoms], weights, bias)
        return tf.nn.softmax(dense3)

    # define a function called build_categorical_DQN for building the main and
    # target categorical deep Q networks
    def build_categorical_DQN(self):

        # define the main categorical DQN and obtain the probabilities
        with tf.variable_scope('main_net'):
            name = ['main_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            weights = tf.random_uniform_initializer(-0.1, 0.1)
            bias = tf.constant_initializer(0.1)

            self.main_p = self.build_network(self.state_ph,
                                             self.action_ph,
                                             name,
                                             24,
                                             24,
                                             weights,
                                             bias)

        # define the target categorical DQN and obtain the probabilities
        with tf.variable_scope('target_net'):
            name = ['target_net_params' ,tf.GraphKeys.GLOBAL_VARIABLES]

            weights = tf.random_uniform_initializer(-0.1, 0.1)
            bias = tf.constant_initializer(0.1)

            self.target_p = self.build_network(self.state_ph,
                                               self.action_ph,
                                               name,
                                               24,
                                               24,
                                               weights,
                                               bias)

        # compute the main Q value with probabilities obtained from the main categorical DQN
        self.main_Q = tf.reduce_sum(self.main_p * self.z)

        # compute the target Q value with probabilities obtained from the target categorical DQN
        self.target_Q = tf.reduce_sum(self.target_p * self.z)

        # define the cross entropy loss
        self.cross_entropy_loss = -tf.reduce_sum(self.m_ph * tf.log(self.main_p))

        # define the optimizer and minimize the cross entropy loss using Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy_loss)

        # get the main network parameters
        main_net_params = tf.get_collection("main_net_params")

        # get the target network parameters
        target_net_params = tf.get_collection('target_net_params')

        # define the update_target_net operation for updating the target network parameters by
        # copying the parameters of the main network
        self.update_target_net = [tf.assign(t, e) for t, e in zip(target_net_params, main_net_params)]


    def train(self, s, r, action, s_, gamma):

        self.time_step += 1

        # target Q values
        list_q_ = [self.sess.run(self.target_Q,
                                 feed_dict = {self.state_ph :[s_], self.action_ph :[[a]]}) for a in range(self.action_shape)]

        # next state action a dash as the one which has the maximum Q value
        a_ = tf.argmax(list_q_).eval()

        # the distributed probability of the target distribution after the projection step
        m = np.zeros(self.atoms)

        # get the probability for each atom using the target categorical DQN
        p = self.sess.run(self.target_p,
                          feed_dict = {self.state_ph :[s_], self.action_ph :[[a_]]})[0]

        for j in range(self.atoms):
            Tz = min(self.v_max, max(self.v_min, r +gamma * self.z[j]))
            bj = (Tz - self.v_min) / self.delta_z
            l, u = math.floor(bj), math.ceil(bj)

            pj = p[j]

            m[int(l)] += pj * (u - bj)
            m[int(u)] += pj * (bj - l)

        self.sess.run(self.optimizer,
                      feed_dict = {self.state_ph :[s], self.action_ph :[action], self.m_ph: m })

        if self.time_step % self.update_target_net == 0:
            self.sess.run(self.update_target_net)


    def select_action(self, s):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_shape - 1)
        else:
            return np.argmax([self.sess.run(self.main_Q,
                                            feed_dict = {self.state_ph :[s], self.action_ph :[[a]]}) for a in range(self.action_shape)])



    def conv(self,
             inputs,
             kernel_shape,
             bias_shape,
             strides,
             weights,
             bias = None,
             activation = tf.nn.relu):

        weights = tf.get_variable('weights', shape = kernel_shape, initializer = weights)
        conv = tf.nn.conv2d(inputs, weights, strides = strides, padding = 'SAME')

        if bias_shape is not None:
            biases = tf.get_variable('biases', shape = bias_shape, initializer = bias)
            conv = activation(conv + biases) if activation is not None else conv + biases
            return conv

        conv = activation(conv) if activation is not None else conv

        return conv

    def dense(self,
              inputs,
              units,
              bias_shape,
              weights,
              bias = None,
              activation = tf.nn.relu):

        if not isinstance(inputs, ops.Tensor):
            inputs = ops.convert_to_tensor(inputs, dtype = 'float')
        if len(inputs.shape) > 2:
            inputs = tf.layers.flatten(inputs)

        flatten_shape = inputs.shape[1]
        weights = tf.get_variable('weights', shape = [flatten_shape, units], initializer = weights)

        dense = tf.matmul(inputs, weights)

        if bias_shape is not None:
            assert bias_shape[0] == units
            biases = tf.get_variable('biases', shape = bias_shape, initializer = bias)

            dense = activation(dense + biases) if activation is not None else dense + biases
            return dense


        dense = activation(dense) if activation is not None else dense

        return dense


    def sample_transitions(self):
        # returns the randomly sampled minibatch of transitions from the replay buffe

        batch = np.random.permutation(len(self.replay_buffer))[: self.batch_size]
        trans = np.array(self.replay_buffer)[batch]

        return trans


if __name__ == '__main__':

    env = gym.make("Tennis-v0")

    agent = Categorical_DQN(env)

    num_episodes = 800
    for i in range(num_episodes):

        done = False

        state = env.reset()

        Return = 0

        while not done:

            env.render()

            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            Return = Return + reward

            agent.replay_buffer.append([state, reward, [action], next_state])

            # if the length of the replay buffer is greater than or equal to buffer size then start training the
            # network by sampling transitions from the replay buffer
            if len(agent.replay_buffer) >= agent.batch_size:
                trans = agent.sample_transitions(2)
                for item in trans:
                    agent.train(item[0], item[1], item[2], item[3], agent.gamma)

            state = next_state

        print("Episode:{}, Return: {}".format(i, Return))

