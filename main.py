"""The main class, for training and testing agents on the environment.
"""

import gym
from agent import Agent, StochasticAgent
import train

env = gym.make('CartPole-v1')
observation = env.reset()

agent = train.reinforce(StochasticAgent())

print(train.avg_reward(agent))
