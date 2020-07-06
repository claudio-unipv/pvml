import numpy as np
from .checks import _check_size, _check_labels


def svm_inference(X, w, b):
    """SVM prediction of the class labels.

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


def svm_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary SVM classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
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
    _check_size("mn, m, n?, *", X, Y, init_w, init_b)
    Y = _check_labels(Y, 2)
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    C = (2 * Y) - 1
    for step in range(steps):
        labels, logits = svm_inference(X, w, b)
        hinge_diff = -C * ((C * logits) < 1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def hinge_loss(labels, logits):
    """Average hinge loss.

    Parameters
    ----------
    labels : ndarray, shape (m,)
        binary target labels (0 or 1).
    logits : ndarray, shape (m,)
        classification scores (logits).

    Returns
    -------
    float
        average hinge loss.
    """
    _check_size("m, m", labels, logits)
    labels = _check_labels(labels, 2)
    loss = np.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()
