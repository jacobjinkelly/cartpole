"""This module records statistics of the agents' performance.
"""

import csv

import numpy as np

import train


def random_trial_length():
    """Comparing trial lengths to likelihood of "random" training algorithm
    finding agent which solves environment.
    """
    vals = [1, 2, 3]
    results = np.zeros((3, 10))
    for i in range(3):
        total = 0
        for j in range(10):
            print(i, j)
            results[i][j] = train.get_avg_reward(train.random(10000, vals[i], 0, 1), 100)
    print(results)
    np.savetxt("results/random_trial_length.csv", results)


def hill_climb_trial_length():
    """Comparing trial lengths to convergence of "hill climbing" algorithm.
    """
    vals = [3, 5, 10, 25, 50, 100]
    with open("results/hill_climb_trial_length.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            trial_length = vals[i]
            writer.writerow([str(trial_length)])
            for j in range(10):
                print(i, j)
                agent, _ = train.hill_climb(trial_length, 0, 1, 200)
                writer.writerow([str(train.get_avg_reward(agent, 100))])


def hill_climb_std_dev():
    """Tuning std_dev hyperparameter of "hill climbing" algorithm.
    """
    vals = [0.1, 0.3, 1]
    with open("results/hill_climb_std_dev.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            std_dev = vals[i]
            writer.writerow([str(std_dev)])
            for j in range(10):
                print(std_dev, j)
                agent, q = train.hill_climb(5, 0, std_dev, 200)
                while True:
                    try:
                        t, reward = q.popleft()
                        writer.writerow((str(t), str(reward)))
                    except IndexError:
                        break
                writer.writerow([str(train.get_avg_reward(agent, 100))])


def reinforce_alpha():
    """Tuning step size of reinforce algorithm.
    """
    vals = [0.0001, 0.001, 0.01, 0.1]
    with open("results/reinforce_alpha.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            alpha = vals[i]
            writer.writerow([str(alpha)])
            for j in range(1):
                print(alpha, j)
                agent, q = train.reinforce(alpha, 5, 200, 200)
                while True:
                    try:
                        t, reward = q.popleft()
                        writer.writerow((str(t), str(reward)))
                    except IndexError:
                        break
                writer.writerow([str(train.get_avg_reward(agent, 100))])


def reinforce_rollouts():
    """Tune number of rollouts.
    """
    vals = [10, 25, 50, 100]
    with open("results/reinforce_rollouts.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            num_rollouts = vals[i]
            writer.writerow([str(num_rollouts)])
            for j in range(1):
                print(num_rollouts, j)
                agent, q = train.reinforce(0.0001, num_rollouts, 200, 200)
                while True:
                    try:
                        t, reward = q.popleft()
                        writer.writerow((str(t), str(reward)))
                    except IndexError:
                        break
                writer.writerow([str(train.get_avg_reward(agent, 100))])


def reinforce_td_alpha():
    """Tuning step size of reinforce algorithm.
    """
    vals = [0.001, 0.003, 0.01, 0.03]
    with open("results/reinforce_td_alpha.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            alpha = vals[i]
            writer.writerow([str(alpha)])
            for j in range(10):
                print(alpha, j)
                agent, q = train.reinforce_td(alpha, 5, 200, 200)
                while True:
                    try:
                        t, reward = q.popleft()
                        writer.writerow((str(t), str(reward)))
                    except IndexError:
                        break
                writer.writerow([str(train.get_avg_reward(agent, 100))])


def reinforce_td_rollouts():
    """Tune number of rollouts.
    """
    vals = [10, 25, 50, 100]
    with open("results/reinforce_td_rollouts.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(vals)):
            num_rollouts = vals[i]
            writer.writerow([str(num_rollouts)])
            for j in range(1):
                print(num_rollouts, j)
                agent, q = train.reinforce_td(0.001, num_rollouts, 200, 200)
                while True:
                    try:
                        t, reward = q.popleft()
                        writer.writerow((str(t), str(reward)))
                    except IndexError:
                        break
                writer.writerow([str(train.get_avg_reward(agent, 100))])
