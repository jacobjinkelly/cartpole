"""The main class, for training and testing agents on the environment.
"""

import gym
from agent import Agent
import train

env = gym.make('CartPole-v0')
observation = env.reset()

agent = train.random(Agent())

for t in range(100):
    env.render()
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
