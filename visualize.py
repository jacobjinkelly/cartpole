"""
This module renders visualizations of statistics obtained from experiment.py
"""

import os

import matplotlib.pyplot as plt


def show_plot(points, save: bool=False, name: str="plot", xlabel: str="", ylabel: str=""):
    """
    Show a scatter plot of the given points.
    """
    x, y = zip(points)

    plt.xticks(x), plt.xlabel(xlabel), plt.ylabel(ylabel)  # formatting
    plt.plot(x, y)
    if save:
        if not os.path.isdir("imgs"):
            os.mkdir("imgs")
        plt.savefig("imgs/" + name + ".png")
    plt.show()
