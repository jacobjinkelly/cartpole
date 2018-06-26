"""This module contains methods for training agents.
"""

from collections import deque
from typing import Tuple, List, Deque

import gym
import numpy as np
from scipy.special import expit

from agent import Agent, StochasticAgent


def random(population: int, num_rollouts: int, mean: float, std_dev: float) -> Agent:
    """Initialize <population> agents randomly, picks the best one.
    The 'best' agent corresponds to the agent with the highest average reward
    over <num_rollouts> trials.
    """
    max_reward = 0
    best_agent = None
    for _ in range(population):
        agent = Agent()
        reward = get_avg_reward(agent, num_rollouts)
        if max_reward < reward:
            best_agent = agent
            max_reward = reward
    return best_agent


def hill_climb(num_rollouts: int, mean: float, std_dev: float, max_reward: int) \
        -> Tuple[Agent, Deque[Tuple[int, float]]]:
    """Initialize an agent randomly, and randomly pertube the weights. If the
    random pertubation achieves better performance, update the weights.
    Hyperparameters:
    num_rollouts: number of trials to sample for avg_reward of agent.
    mean: mean of gaussian which weight pertubations are sampled from.
    std_dev: std dev of gaussian which weight pertubations are sampled from.
    max_reward: train the agent until it achieves <max_reward>
    """
    agent = Agent()
    q = deque()
    t = 0
    reward = get_avg_reward(agent, num_rollouts)
    while reward < max_reward:
        q.append((t, reward))
        perturb = std_dev * np.random.randn(4) + mean
        agent.weights += perturb
        if get_avg_reward(agent, num_rollouts) <= reward:
            agent.weights -= perturb
        else:
            reward = get_avg_reward(agent, num_rollouts)
        t += 1
    return agent, q


def reinforce(alpha: float, num_rollouts: int, horizon: int, max_reward: float) \
        -> Tuple[StochasticAgent, Deque[Tuple[int, float]]]:
    """Trains an agent with a stochastic policy (<agent>) using the standard
    REINFORCE policy gradient algorithm.
    Hyperparameters:
    alpha: step size
    num_rollouts: number of rollouts to sample
    horizon: time horizon for rollout
    max_reward: train the agent until it achieves <max_reward>
    """
    agent = StochasticAgent()
    q = deque()
    t = 0
    reward, data = sample_rollout(agent, num_rollouts, horizon)
    while reward < max_reward:
        q.append((t, reward))
        grad = np.zeros(4)
        for i in range(len(data)):
            state, action = data[i]
            z = np.dot(agent.weights, state)  # activation
            sigmoid = expit(z)
            grad += (action * (1 - sigmoid) + (action - 1) * sigmoid) * state

        agent.weights += alpha * reward * grad
        reward, data = sample_rollout(agent, num_rollouts, horizon)
        t += 1
        if t > 1000:
            q.append((t, -1))
            break

    return agent, q


def reinforce_td(alpha: float, num_rollouts: int, horizon: int, max_reward: float) -> Tuple[StochasticAgent, deque]:
    """Trains an agent with a stochastic policy (<agent>) using modified
    (temporal difference) REINFORCE policy gradient algorithm.
    Hyperparameters:
    alpha: step size
    num_rollouts: number of rollouts to sample
    horizon: time horizon for rollout
    max_reward: train the agent until it achieves <max_reward>
    """
    agent = StochasticAgent()
    q = deque()
    t = 0
    prev_reward = 0
    reward, data = sample_rollout(agent, num_rollouts, horizon)
    while reward < max_reward:
        q.append((t, reward))
        grad = np.zeros(4)
        for i in range(len(data)):
            state, action = data[i]
            z = np.dot(agent.weights, state)  # activation
            sigmoid = expit(z)
            grad += (action * (1 - sigmoid) + (action - 1) * sigmoid) * state

        agent.weights += alpha * (reward - prev_reward) * grad
        prev_reward, (reward, data) = reward, sample_rollout(agent,
                                                             num_rollouts, horizon)
        t += 1
        if t > 1000:
            q.append((t, -1))
            break

    return agent, q


def sample_rollout(agent: Agent, num_rollouts: int, horizon: int) \
        -> Tuple[float, List[Tuple[np.ndarray, int]]]:
    """Samples <num_rollouts> rollouts with time horizon <horizon>, and returns
    a tuple (avg reward, List(state, action))
    """

    env = gym.make('CartPole-v0')

    cumulative_reward = 0
    state_action = []

    for i in range(num_rollouts):
        t = 0
        done = False

        observation = env.reset()

        while (t < horizon) and not done:
            action = agent.get_action(observation)
            state_action.append((observation, action))
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            t += 1

    return (cumulative_reward / num_rollouts, state_action)


def get_reward(agent: Agent) -> int:
    """Returns the cumulative reward gained by <agent> in one episode in the
    training environment.
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
    """Returns the average cumulative reward over <num_trials> trials.
    """
    total = 0
    for _ in range(num_trials):
        total += get_reward(agent)
    return total / num_trials


def render(agent: Agent) -> None:
    """Renders <agent> interacting with <env> to the screen.
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
