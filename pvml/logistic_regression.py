import numpy as np


def logreg_inference(X, w, b):
    """Predict class probabilities.

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
        probability estimates (one per feature vector).
    """
    logits = X @ w + b
    return 1 / (1 + np.exp(-logits))


def binary_cross_entropy(Y, P):
    """Average cross entropy.

    Parameters
    ----------
    Y : ndarray, shape (m,)
        binary target labels (0 or 1).
    P : ndarray, shape (m,)
        probability estimates.

    Returns
    -------
    float
        average cross entropy.
    """
    eps = 1e-3
    P = np.clip(P, eps, 1 - eps)  # This prevents overflows
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean()


def logreg_train(X, Y, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary classifier based on logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
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
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = ((P - Y) @ X) / m
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def logreg_l2_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None,
                    init_b=0):
    """Train a binary classifier based on L2-regularized logistic regression.

    Parameters
    ----------

    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
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
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = ((P - Y) @ X) / m + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def logreg_l1_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary classifier based on L1-regularized logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
    loss : ndarray, shape (steps,)
        loss value after each training step.

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    """
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = ((P - Y) @ X) / m + lambda_ * np.sign(w)
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
