"""
This module records statistics of the agents' performance.
"""

from typing import List

import train
from visualize import show_plot


def random_num_trials(nums_trials: List[int], num_samples: int):
    """
    Comparing number of trials to performance of "random" training algorithm at finding agent which solves environment.
    """
    results = []
    for num_trials in nums_trials:
        for i in range(num_samples):
            print("num_trials: %d, sample: %d" % (num_trials, i + 1))
            results.append((num_trials, train.get_avg_reward(agent=train.random(population=10000,
                                                                                num_trials=num_trials,
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
    for num_trials in nums_trials:
        for i in range(num_samples):
            print("num_trials: %d, sample: %d" % (num_trials, i + 1))
            agent, _ = train.hill_climb(num_trials=num_trials,
                                        mean=0,
                                        std_dev=1,
                                        max_reward=200)
            results.append((num_trials, train.get_avg_reward(agent=agent, num_trials=100)))
    show_plot(results, save=True, name="hill_climb_trial_length", xlabel="Number of Trials",
              ylabel="Avg. Reward over 100 trials")


def hill_climb_std_dev(std_devs: List[int], num_samples: int):
    """
    Comparing standard deviation of distribution from which weight pertubation is sampled from to performance of hill
    climbing training algorithm at finding agent which solves environment.
    """
    results_std_dev, results_trajectories = [], {std_dev: [] for std_dev in std_devs}
    for std_dev in std_devs:
        for i in range(num_samples):
            print("std_dev: %d, sample: %d" % (std_dev, i + 1))
            agent, trajectory = train.hill_climb(num_trials=5,
                                                 mean=0,
                                                 std_dev=std_dev,
                                                 max_reward=200)
            results_trajectories[std_dev].append(trajectory)
            results_std_dev.append((std_dev, train.get_avg_reward(agent=agent, num_trials=100)))


def reinforce_lr(lrs: List[float], num_samples: int):
    """
    Comparing learning rate to performance of REINFORCE algorithm at finding agent which solves environment.
    """
    results_lr, results_trajectories = [], {lr: [] for lr in lrs}
    for lr in lrs:
        for i in range(num_samples):
            print("learning rate: %s, sample: %d" % (lr, i + 1))
            agent, trajectory = train.reinforce(lr=lr,
                                                num_rollouts=5,
                                                horizon=200,
                                                max_reward=200)
            results_trajectories[lr].append(trajectory)
            results_lr.append((lr, train.get_avg_reward(agent=agent, num_trials=100)))


def reinforce_rollouts(nums_rollouts: List[int], num_samples: int):
    """
    Comparing number of rollouts to performance of REINFORCE algorithm at finding agent which solves environment.
    """
    results_rollouts, results_trajectories = [], {num_rollouts: [] for num_rollouts in nums_rollouts}
    for num_rollouts in nums_rollouts:
        for i in range(num_samples):
            print("num_rollouts: %d, sample: %d" % (num_rollouts, i + 1))
            agent, trajectory = train.reinforce(lr=0.0001,
                                                num_rollouts=num_rollouts,
                                                horizon=200,
                                                max_reward=200)
            results_trajectories[num_rollouts].append(trajectory)
            results_rollouts.append((num_rollouts, train.get_avg_reward(agent=agent, num_trials=100)))
