import numpy as np


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
    w = np.zeros(X.shape[1])
    b = 0
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
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits
