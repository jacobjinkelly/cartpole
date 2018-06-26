"""
Miscellaneous utilities.
"""

np_seed = None


def set_np_seed(seed):
    """
    Set seed for np.random across entire project.
    """
    global np_seed
    np_seed = seed
