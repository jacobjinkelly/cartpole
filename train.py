"""This module contains methods for training agents.
"""

from agent import Agent
import gym
import numpy as np

def random(agent: Agent) -> Agent:
    """Initialize 10,000 agents randomly, and pick the best one.
    """
    max_reward = 0
    best_agent = None
    for _ in range(10000):
        if (max_reward < reward(agent)):
            best_agent = agent
    return best_agent

def hill_climb(agent: Agent) -> Agent:
    """Initialize an agent randomly, and randomly pertube the weights. If the
    random pertubation achieves better performance, update the weights.
    """
    agent = Agent()
    r = avg_reward(agent)
    while (r < 195):
        pertub = 0.1 * np.random.randn(4)
        agent.set_weights(agent.weights + pertub)
        if (avg_reward(agent) <= r):
            agent.set_weights(agent.weights - pertub)
        else:
            r = avg_reward(agent)
    return agent

def reward(agent: Agent) -> int:
    """Returns the cumulative reward gained by <agent> in one episode in the
    training environment.
    """
    env = gym.make('CartPole-v1')
    observation = env.reset()

    cumulative_reward = 0
    t = 0
    while True:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        t += 1

    return cumulative_reward

def avg_reward(agent: Agent) -> float:
    """Returns the average cumulative reward over 100 trials.
    """
    sum = 0
    for _ in range(100):
        sum += reward(agent)
    return sum/100.0
