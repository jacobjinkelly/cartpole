"""
The main class, for training and testing agents on the environment.
"""

import gym

import experiment

# set level to ERROR so doesn't log WARN level (in particular so it doesn't WARN about automatic detecting of dtype)
gym.logger.set_level(40)

experiment.reinforce_td_alpha()
