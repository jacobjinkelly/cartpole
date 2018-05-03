"""This module records statistics of the agents' performance.
"""

from __future__ import division # force float division
import train
import numpy as np
import csv

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

def hill_climb_std_dev():
    """Comparing std_dev to rate of convergence of "hill climbing" algorithm.
    """
    vals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    with open("results/hill_climb_std_dev.csv", "w", newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ' ', quotechar = '|',
                                                    quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            std_dev = 1
            writer.writerow(str(std_dev))
            _, q = train.hill_climb(3, 0, std_dev, 200)
            while True:
                try:
                    t, reward = q.popleft()
                    writer.writerow((str(t), str(reward)))
                except IndexError :
                    break
