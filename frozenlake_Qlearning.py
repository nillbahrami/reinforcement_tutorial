import gym
import numpy as np
import random

def play():
    env = gym.make('FrozenLake-v1')

    action_size = env.action_space.n
    state_size = env.observation_space.n

    qtable = np.zeros((state_size, action_size))

    print('Q table: ', qtable)

    total_episodes = 10000
    learning_rate = 0.8  # alpha
    max_steps = 99
    gamma = 0.95

    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    rewards = []

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_reward = 0

        for step in range(max_steps):

            exp_tradeoff = random.uniform(0, 1)

            if exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # Q function
            qtable[state, action] = qtable[state, action] + learning_rate * (
                        reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            total_reward += reward

            state = new_state

            if done:
                break

        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        rewards.append(total_reward)

    print("Score over time: " + str(sum(rewards) / total_episodes))
    print(qtable)

    env.reset()

    for episode in range(5):
        state = env.reset()
        step = 0
        done = False
        print("======================================================")
        print('Episode: ', episode)

        for step in range(max_steps):
            env.render()
            action = np.argmax(qtable[state, :])
            new_state, reward, done, info = env.step(action)
            if done:
                break
            state = new_state

            print('reward of step: ', reward)

    env.close()