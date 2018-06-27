"""This module contains methods for training agents.
"""

from typing import Tuple, List

import gym
import numpy as np
from scipy.special import expit

from agent import Agent, StochasticAgent
from utils import np_seed

np.random.seed(np_seed)


def random(population: int, num_trials: int, mean: float, std_dev: float) -> Agent:
    """
    Initialize <population> agents randomly, picks the best one.
    The 'best' agent corresponds to the agent with the highest average reward
    over <num_trials> trials.
    """
    max_reward = 0
    best_agent = None
    for _ in range(population):
        agent = Agent(mean=mean, std_dev=std_dev)
        reward = get_avg_reward(agent, num_trials)
        if max_reward < reward:
            best_agent, max_reward = agent, reward
    return best_agent


def hill_climb(num_trials: int, mean: float, std_dev: float, max_reward: int) -> Tuple[Agent, List[Tuple[int, float]]]:
    """
    Initialize an agent randomly, and randomly pertube the weights. If the random pertubation achieves better
    performance, update the weights.
    Hyperparameters:
    num_trials: number of trials to sample for avg_reward of agent.
    mean: mean of gaussian which weight pertubations are sampled from.
    std_dev: std dev of gaussian which weight pertubations are sampled from.
    max_reward: train the agent until it achieves <max_reward>
    """
    agent = Agent()
    trajectory = []
    t, reward = 0, get_avg_reward(agent, num_trials)
    while reward < max_reward:
        trajectory.append((t, reward))
        perturb = std_dev * np.random.randn(4) + mean
        agent.weights += perturb
        if get_avg_reward(agent, num_trials) <= reward:
            agent.weights -= perturb
        else:
            reward = get_avg_reward(agent, num_trials)
        t += 1
    return agent, trajectory


def reinforce(lr: float, num_trials: int, horizon: int, max_reward: float) \
        -> Tuple[StochasticAgent, List[Tuple[int, float]]]:
    """
    Trains an agent with a stochastic policy (<agent>) using the standard REINFORCE policy gradient algorithm.
    Hyperparameters:
    lr: learning rate
    num_trials: number of trials to sample
    horizon: time horizon for each trial
    max_reward: train the agent until it achieves <max_reward>
    """
    agent = StochasticAgent()
    trajectory = []
    t = 0
    reward, data = sample_trials(agent, num_trials, horizon)
    while reward < max_reward:
        trajectory.append((t, reward))
        grad = np.zeros(4)
        for datum in data:
            state, action = datum
            z = np.dot(agent.weights, state)  # activation
            sigmoid = expit(z)
            grad += (action * (1 - sigmoid) + (action - 1) * sigmoid) * state

        agent.weights += lr * reward * grad
        reward, data = sample_trials(agent, num_trials, horizon)
        t += 1
        if t > 1000:
            trajectory.append((t, -1))
            break

    return agent, trajectory


def sample_trials(agent: Agent, num_trials: int, horizon: int) -> Tuple[float, List[Tuple[np.ndarray, int]]]:
    """
    Samples <num_trials> trials with time horizon <horizon>, and returns a tuple (avg reward, List(state, action))
    """
    env = gym.make('CartPole-v0')

    cumulative_reward = 0
    state_action = []

    for i in range(num_trials):
        t = 0
        done = False

        observation = env.reset()

        while (t < horizon) and not done:
            action = agent.get_action(observation)
            state_action.append((observation, action))
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            t += 1

    return (cumulative_reward / num_trials, state_action)


def get_reward(agent: Agent) -> int:
    """
    Returns the cumulative reward gained by <agent> in one episode in the training environment.
    """
    env = gym.make('CartPole-v0')
    observation = env.reset()

    cumulative_reward = 0
    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break

    return cumulative_reward


def get_avg_reward(agent: Agent, num_trials: int) -> float:
    """
    Returns the average cumulative reward over <num_trials> trials.
    """
    total = 0
    for _ in range(num_trials):
        total += get_reward(agent)
    return total / num_trials


def render(agent: Agent) -> None:
    """
    Renders <agent> interacting with <env> to the screen.
    """
    env = gym.make("CartPole-v0")
    observation = env.reset()

    t = 0
    while True:
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        t += 1
