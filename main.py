"""The main class, for training and testing agents on the environment.
"""

import gym
from agent import Agent, StochasticAgent
import train

gym.logger.set_level(40) # set level to ERROR, i.e. so doesn't log WARN level
# (in particular so it doesn't WARN about automatic detecting of dtype)

agent = train.reinforce(StochasticAgent())

print(train.avg_reward(agent))
