import numpy as np
from .checks import _check_size, _check_labels


def perceptron_train(X, Y, steps=10000, init_w=None, init_b=0):
    """Train a binary classifier using the perceptron algorithm.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    steps: int
        maximum number of training iterations
    init_w : ndarray, shape (n,)
        initial weights (None for zero initialization)
    init_b : float
        initial bias

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    """
    _check_size("mn, m, n?, *?", X, Y, init_w, init_b)
    Y = _check_labels(Y)
    w = (init_w if init_w is not None else np.zeros(X.shape[1]))
    b = init_b
    for step in range(steps):
        errors = 0
        for i in range(X.shape[0]):
            d = (1 if X[i, :] @ w + b > 0 else 0)
            w += (Y[i] - d) * X[i, :].T
            b += (Y[i] - d)
            errors += np.abs(Y[i] - d)
        if errors == 0:
            break
    return w, b


def perceptron_inference(X, w, b):
    """Perceptron prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    w : ndarray, shape (n,)
         weight vector.
    b : float
         scalar bias.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m,)
        classification scores (one per feature vector).
    """
    _check_size("mn, n, *", X, w, b)
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits
