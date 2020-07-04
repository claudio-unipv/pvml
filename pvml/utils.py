import numpy as np


def log_nowarn(x):
    """Compute the logarithm without warnings in case of zeros."""
    with np.errstate(divide='ignore'):
        return np.log(x)


def one_hot_vectors(Y, classes):
    m = Y.shape[0]
    H = np.zeros((m, classes), dtype=int)
    H[np.arange(m), Y] = 1
    return H
