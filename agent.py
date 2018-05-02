"""This module contains the agent classes.
"""

import numpy as np
from scipy.special import expit

class Agent:
    """An agent in the environment.

    === Attributes ===
    weights:
        The weights used in the agent's linear model of the environment.
    """
    weights: np.ndarray

    def __init__(self, MEAN: float = 0, STD_DEV: float = 1) -> None:
        """Initalize weights over Gaussian with <MEAN> and <STD_DEV>.
        """
        self.weights = STD_DEV * np.random.randn(4) + MEAN

    def get_action(self, obs: np.ndarray) -> int:
        """Get the agent's next action, given an observation <obs>.
        """
        if (np.dot(self.weights, obs)) >= 0:
            return 1
        else:
            return 0

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the agents weights to <weights>.
        """
        self.weights = weights

class StochasticAgent(Agent):
    """An agent with a stochastic policy in the environment.
    """

    def get_action(self, obs: np.ndarray) -> int:
        prob = expit(np.dot(self.weights, obs))
        return np.random.choice([0, 1], p = [1 - prob, prob])
