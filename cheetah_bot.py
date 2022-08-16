import gym

# import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2


import imageio
import numpy as np

def run_cheetah_run():

    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v2")])

    env = VecNormalize(env, norm_obs = True)
    agent = PPO2(MlpPolicy, env)

    agent.learn(total_timesteps = 250000)

    state = env.reset()
    while True:
        action, _ = agent.predict(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        env.render()

    images = []
    state = agent.env.reset()
    img = agent.env.render(mode = 'rgb_array')
    for i in range(500):
        images.append(img)
        action, _ = agent.predict(state)
        next_state, reward, done, info = agent.env.step(action)
        state = next_state
        img = agent.env.render(mode = 'rgb_array')


    imageio.mimsave('CheetahCanRun.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps = 29)
    
    
if __name__ == '__main__':
    
    run_cheetah_run()    




