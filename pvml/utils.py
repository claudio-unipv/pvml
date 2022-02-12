import numpy as np


def log_nowarn(x):
    """Compute the logarithm without warnings in case of zeros."""
    with np.errstate(divide='ignore'):
        return np.log(x)


def one_hot_vectors(Y, classes):
    """Create the array with the one-hot-vector representation of class labels in Y."""
    Y = np.asarray(Y).astype(int)
    m = Y.shape[0]
    H = np.zeros((m, classes), dtype=int)
    H[np.arange(m), Y] = 1
    return H


def squared_distance_matrix(X1, X2):
    """Compute the matrix D.

    D[i, j] is the square of the distance between X1[i, :] and X2[j ,:].
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Q1 = (X1 ** 2).sum(1, keepdims=True)
    Q2 = (X2 ** 2).sum(1, keepdims=True)
    return Q1 - 2 * X1 @ X2.T + Q2.T
