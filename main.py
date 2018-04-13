"""The main class, for training and testing agents on the environment.
"""

import gym
from agent import Agent
import train

env = gym.make('CartPole-v1')
observation = env.reset()

agent = train.hill_climb(Agent())
