"""
This module contains the agent classes.
"""

import numpy as np
from scipy.special import expit

from utils import np_seed

np.random.seed(np_seed)


class Agent:
    """
    An agent in the environment.

    === Attributes ===
    weights:
        The weights used in the agent's linear model of the environment.
    """
    weights: np.ndarray

    def __init__(self, mean: float=0, std_dev: float=1) -> None:
        """
        Initialize weights over Gaussian with <mean> and <std_dev>.
        """
        self.weights = std_dev * np.random.randn(4) + mean

    def get_action(self, obs: np.ndarray) -> int:
        """
        Get the agent's next action, given an observation <obs>.
        """
        if (np.dot(self.weights, obs)) >= 0:
            return 1
        else:
            return 0

    def init_weights(self, mean: float=0, std_dev: float=1) -> None:
        """
        Re-initialize the weights of the agent over Gaussian with <mean> and <std_dev>.
        """
        self.weights = std_dev * np.random.rand(4) + mean

    def get_weights(self) -> np.ndarray:
        """
        Get the agent weights.
        """
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the agents weights to <weights>.
        """
        self.weights = weights


class StochasticAgent(Agent):
    """
    An agent with a stochastic policy in the environment.
    """

    def get_action(self, obs: np.ndarray) -> int:
        """
        Sample from the agent's distribution of actions, given an observation <obs>.
        """
        prob = expit(np.dot(self.weights, obs))
        return np.random.choice([0, 1], p=[1 - prob, prob])
