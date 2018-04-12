"""This module contains the agent classes.
"""

import numpy as np

class Agent:
    """An agent in the environment.

    This is an abstract class. Only child classes should be instantiated.

    === Attributes ===
    weights:
        The weights used in the agent's linear model of the environment.
    """
    weights: np.ndarray

    def __init__(self) -> None:
        """Initalize weights to be a random np.ndarray of dimension 4.
        """
        self.weights = np.random.rand(4)

    def get_action(self, obs: np.ndarray) -> None:
        """Get the agent's next action, given an observation obs.
        """
        if (np.dot(self.weights, obs)) >= 0:
            return 1
        else:
            return 0
