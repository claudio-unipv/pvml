import numpy as np
from .svm import svm_train
from .checks import _check_size, _check_labels


def one_vs_one_svm_inference(X, W, b):
    """Multiclass linear SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    W : ndarray, shape (n, s)
         weights (one vector per pair of classes).
    b : ndarray, shape (s,)
         vector of biases.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector) in the range 0...(k-1).
    ndarray, shape (m, k)
        classification scores.
    """
    _check_size("mn, ns, s", X, W, b)
    # 1) recover the number of classes from s = 1 + 2 + ... + k
    m = X.shape[0]
    s = b.size
    k = int(1 + np.sqrt(1 + 8 * s)) // 2
    votes = np.zeros((m, k))
    logits = X @ W + b.T
    bin_labels = (logits > 0)
    # For each pair of classes...
    j = 0
    for pos in range(k):
        for neg in range(pos + 1, k):
            votes[:, pos] += bin_labels[:, j]
            votes[:, neg] += (1 - bin_labels[:, j])
            j += 1
    labels = np.argmax(votes, 1)
    return labels, votes


def one_vs_one_svm_train(X, Y, lambda_, lr=1e-3, steps=1000,
                         init_w=None, init_b=None):
    """Train a multi-class linear SVM using the one vs. one strategy.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels in the range 0, ..., (k - 1).
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_w : ndarray, shape (n, k * (k - 1) // 2)
        initial weights (None for zero initialization)
    init_b : ndarray, shape (k * (k - 1) // 2,)
        initial biases (None for zero initialization)

    Returns
    -------
    W : ndarray, shape (n, k * (k - 1) // 2)
        learned weights (one vector per each pair of classes).
    b : ndarray, shape (k * (k - 1) // 2,)
        vector of biases.
    """
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    W = np.zeros((n, k * (k - 1) // 2))
    b = np.zeros(k * (k - 1) // 2)
    j = 0
    # For each pair of classes...
    for pos in range(k):
        for neg in range(pos + 1, k):
            # Build a training subset
            subset = (np.logical_or(Y == pos, Y == neg)).nonzero()[0]
            Xbin = X[subset, :]
            Ybin = (Y[subset] == pos)
            w1 = (None if init_w is None else init_w[:, j])
            b1 = (0 if init_b is None else init_b[j])
            # Train the classifier
            Wbin, bbin = svm_train(Xbin, Ybin, lambda_, lr=lr, steps=steps,
                                   init_w=w1, init_b=b1)
            W[:, j] = Wbin
            b[j] = bbin
            j += 1
    return W, b


def one_vs_rest_svm_inference(X, W, b):
    """Multiclass linear SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    W : ndarray, shape (n, k)
         weights (one vector per class).
    b : ndarray, shape (k,)
         vector of biases.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector) in the range 0...(k-1).
    ndarray, shape (m, k)
        classification scores.
    """
    _check_size("mn, nk, k", X, W, b)
    logits = X @ W + b.T
    labels = np.argmax(logits, 1)
    return labels, logits


def one_vs_rest_svm_train(X, Y, lambda_, lr=1e-3, steps=1000,
                          init_w=None, init_b=None):
    """Train a multi-class linear SVM using the one vs. rest strategy.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels in the range 0, ..., (k - 1).
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_w : ndarray, shape (n, k)
        initial weights (None for zero initialization)
    init_b : ndarray, shape (k,)
        initial biases (None for zero initialization)

    Returns
    -------
    W : ndarray, shape (n, k)
        learned weights (one vector per class).
    b : ndarray, shape (k,)
        vector of biases.
    """
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    W = np.zeros((n, k))
    b = np.zeros(k)
    for c in range(k):
        Ybin = (Y == c)
        w1 = (None if init_w is None else init_w[:, c])
        b1 = (0 if init_b is None else init_b[c])
        Wbin, bbin = svm_train(X, Ybin, lambda_, lr=lr, steps=steps,
                               init_w=w1, init_b=b1)
        W[:, c] = Wbin
        b[c] = bbin
    return W, b
