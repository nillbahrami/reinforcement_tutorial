import gym
import pandas as pd
from collections import defaultdict


class BlackJak:

    def __int__(self):

        self.env = gym.make('Blackjack-v1')

        state = self.env.reset()
        print(state)
        print(self.env.action_space)


    def _policy(self, state):
        return 0 if state[0] > 19 else 1


    def _generate_episode(self, policy):
        episode = []
        state = self.env.reset()

        num_timesteps = 100
        for t in range(num_timesteps):

            action = self._policy(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def play(self):

        print(self._generate_episode(self._policy), self.env)

        # computing the value function
        total_return = defaultdict(float)
        N = defaultdict(int)

        num_iterations = 10000
        for i in range(num_iterations):

            episode = self._generate_episode(self._policy)
            states, actions, rewards = zip(*episode)

            for t, state in enumerate(states):
                R = (sum(rewards[t:]))
                total_return[state] = total_return[state] + R
                N[state] = N[state] + 1  # number of times the state is visited in the episode

        total_return = pd.DataFrame(total_return.items(), columns = ['state', 'total_return'])
        N = pd.DataFrame(N.items(), columns = ['state', 'N'])

        df = pd.merge(total_return, N, on = 'state')

        df['value'] = df['total_return'] / df['N']

        return df



if __name__ == '__main__':

    game = BlackJak()
    game.play()
