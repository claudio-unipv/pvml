import numpy as np


def log_nowarn(x):
    """Compute the logarithm without warnings in case of zeros."""
    with np.errstate(divide='ignore'):
        return np.log(x)
