"""
This module renders visualizations of statistics obtained from experiment.py
"""
import os
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def show_plot(points: List[Tuple[int, float]], save: bool=False, name: str="plot", xlabel: str="", ylabel: str=""):
    """
    Show a scatter plot of the given points.
    """
    x, y = zip(*points)

    plt.xticks(x), plt.xlabel(xlabel), plt.ylabel(ylabel)  # formatting
    plt.scatter(x, y)
    if save:
        if not os.path.isdir("imgs"):
            os.mkdir("imgs")
        plt.savefig("imgs/" + name + ".png")
    plt.show()


def show_freq_hist(vals: List[int], save: bool=False, name: str="plot", xlabel: str="", ylabel: str=""):
    """
    Plots a (normalized) frequency histogram
    """
    plt.xticks([i for i in range(max(vals) + 1)]), plt.xlabel(xlabel), plt.ylabel(ylabel)  # formatting
    # credit to https://stackoverflow.com/questions/9767241/setting-a-relative-frequency-in-a-matplotlib-histogram
    plt.hist(vals, weights=np.zeros_like(vals) + 1 / len(vals))
    if save:
        if not os.path.isdir("imgs"):
            os.mkdir("imgs")
        plt.savefig("imgs/" + name + ".png")
    plt.show()


def draw_function(f: Callable[[np.ndarray], float], save: bool=True, name: str="", clear_plot: bool=True):
    """
    Plot a smooth curve of the 1D function <f>.
    """
    if clear_plot:
        plt.clf()
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, f(x))
    if save:
        if not os.path.isdir("imgs"):
            os.mkdir("imgs")
        plt.savefig("imgs/" + name + ".png")


def draw_step_and_sigmoid():
    """
    Plot a smooth curve of the sigmoid and step function on the same figure.
    """
    draw_function(f=lambda x: 1 / (1 + np.exp(-x)),
                  save=False)
    draw_function(f=lambda x: np.abs(x) / (2 * x) + 1 / 2,
                  save=True,
                  name="step and sigmoid",
                  clear_plot=False)
