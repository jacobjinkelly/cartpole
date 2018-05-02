"""This module records statistics of the agents' performance.
"""

from __future__ import division # force float division
import train
import numpy as np

def random_trial_length():
    """Comparing trial lengths to likelihood of "random" training algorithm
    finding agent which solves environment.
    """
    vals = [1, 2, 3]
    results = np.zeros((3, 10))
    for i in range (3):
        total = 0
        for j in range(10):
            print(i, j)
            results[i][j] = train.avg_reward(train.random(10000, vals[i], 0, 1), 100)
    print(results)
    np.savetxt("results/random_trial_length.csv", results)
