"""
This module records statistics of the agents' performance.
"""

import csv
from typing import List

import train
from visualize import show_plot


def random_num_trials(nums_trials: List[int], num_samples: int):
    """
    Comparing number of trials to performance of "random" training algorithm at finding agent which solves environment.
    """
    results = []
    for i in range(len(nums_trials)):
        for j in range(num_samples):
            print("num_trials: %d, sample: %d" % (nums_trials[i], j + 1))
            results.append((nums_trials[i], train.get_avg_reward(agent=train.random(population=10000,
                                                                                    num_trials=nums_trials[i],
                                                                                    mean=0,
                                                                                    std_dev=1),
                                                                 num_trials=100)))
    show_plot(results, save=True, name="random_trial_length", xlabel="Number of Trials",
              ylabel="Avg. Reward over 100 trials")


def hill_climb_trial_length(nums_trials: List[int], num_samples: int):
    """
    Comparing number of trials to performance of hill climbing training algorithm at finding agent which solves
    environment.
    """
    results = []
    for i in range(len(nums_trials)):
        for j in range(num_samples):
            print("num_trials: %d, sample: %d" % (nums_trials[i], j + 1))
            agent, _ = train.hill_climb(num_trials=nums_trials[i],
                                        mean=0,
                                        std_dev=1,
                                        max_reward=200)
            results.append((nums_trials[i], train.get_avg_reward(agent=agent, num_trials=100)))
    show_plot(results, save=True, name="hill_climb_trial_length", xlabel="Number of Trials",
              ylabel="Avg. Reward over 100 trials")


def hill_climb_std_dev(std_devs: List[int], num_samples: int):
    """
    Tuning std_dev hyperparameter of "hill climbing" algorithm.
    """
    results_std_dev, results_trajectories = [], {std_dev: [] for std_dev in std_devs}
    for i in range(len(std_devs)):
        for j in range(num_samples):
            print("std_dev: %d, sample: %d" % (std_devs[i], j + 1))
            agent, trajectory = train.hill_climb(num_trials=5,
                                                 mean=0,
                                                 std_dev=std_devs[i],
                                                 max_reward=200)
            results_trajectories[std_devs[i]].append(list(trajectory))
            results_std_dev.append((std_devs[i], train.get_avg_reward(agent=agent, num_trials=100)))


def reinforce_alpha():
    """
    Tuning step size of reinforce algorithm.
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
    """
    Tune number of rollouts.
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
    """
    Tuning step size of reinforce algorithm.
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
    """
    Tune number of rollouts.
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
