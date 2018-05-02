"""This module contains methods for training agents.
"""

from __future__ import division # force float division
from agent import Agent, StochasticAgent
import gym
import numpy as np
from scipy.special import expit
from typing import Tuple, List

def random(POPULATION: int, NUM_ROLLOUTS: int) -> Agent:
    """Initialize <POPULATION> agents randomly, picks the best one.
    The 'best' agent corresponds to the agent with the highest average reward
    over <NUM_ROLLOUTS> trials.
    """
    max_reward = 0
    best_agent = None
    for _ in range(POPULATION):
        agent = Agent()
        reward = avg_reward(agent, NUM_ROLLOUTS)
        if (max_reward < reward):
            best_agent = agent
            max_reward = reward
    return best_agent

def hill_climb(NUM_ROLLOUTS: int, MEAN: float, STD_DEV: float,
                                                    MAX_REWARD: int) -> Agent:
    """Initialize an agent randomly, and randomly pertube the weights. If the
    random pertubation achieves better performance, update the weights.
    Hyperparameters:
    NUM_ROLLOUTS: number of trials to sample for avg_reward of agent.
    MEAN: mean of gaussian which weight pertubations are sampled from.
    STD_DEV: std dev of gaussian which weight pertubations are sampled from.
    MAX_REWARD: train the agent until it achieves <MAX_REWARD>
    """
    agent = Agent()
    reward = avg_reward(agent, NUM_ROLLOUTS)
    while (reward < MAX_REWARD):
        pertub = STD_DEV * np.random.randn(4) + MEAN
        agent.set_weights(agent.weights + pertub)
        if (avg_reward(agent, NUM_ROLLOUTS) <= reward):
            agent.set_weights(agent.weights - pertub)
        else:
            reward = avg_reward(agent, NUM_ROLLOUTS)
    return agent

def reinforce(ALPHA: float, NUM_ROLLOUTS: int, HORIZON: int, MAX_REWARD: float)\
                                                            -> StochasticAgent:
    """Trains an agent with a stochastic policy (<agent>) using the standard
    REINFORCE policy gradient algorithm.
    Hyperparameters:
    ALPHA: step size
    NUM_ROLLOUTS: number of rollouts to sample
    HORIZON: time horizon for rollout
    MAX_REWARD: train the agent until it achieves <MAX_REWARD>
    """
    agent = StochasticAgent()
    prev_reward = 0
    reward, data = sample_rollout(agent, NUM_ROLLOUTS, HORIZON)
    while reward < MAX_REWARD:
        grad = np.zeros(4)
        for i in range(len(data)):
            state, action = data[i]
            z = np.dot(agent.weights, state) # activation
            sigmoid = expit(z)
            grad += (action * (1 - sigmoid) + (action - 1) * sigmoid) * state

        agent.weights += ALPHA * (reward - prev_reward) * grad
        prev_reward, (reward, data) = reward, sample_rollout(agent, NUM_ROLLOUTS, HORIZON)

    return agent

def sample_rollout(agent: Agent, NUM_ROLLOUTS: int, HORIZON: int) \
                                -> Tuple[float, List[Tuple[np.ndarray, int]]]:
    """Samples <NUM_ROLLOUTS> rollouts with time horizon <HORIZON>, and returns
    a tuple (avg reward, List(state, action))
    """

    env = gym.make('CartPole-v0')

    cumulative_reward = 0
    state_action = []

    for i in range(NUM_ROLLOUTS):
        t = 0
        done = False

        observation = env.reset()

        while (t < HORIZON) and not done:
            action = agent.get_action(observation)
            state_action.append((observation, action))
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            t += 1

    return (cumulative_reward / NUM_ROLLOUTS, state_action)

def reward(agent: Agent) -> int:
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

def avg_reward(agent: Agent, num_trials: int) -> float:
    """Returns the average cumulative reward over <num_trials> trials.
    """
    sum = 0
    for _ in range(num_trials):
        sum += reward(agent)
    return sum / num_trials

def render(agent: Agent) -> None:
    """Renders <agent> interacting with <env> to the screen.
    """

    env = gym.make('CartPole-v0')
    observation = env.reset()

    t = 0
    while True:
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        t += 1
