import numpy as np
from .ksvm import ksvm_train, kernel
from .checks import _check_size, _check_labels


def one_vs_one_ksvm_inference(X, Xtrain, alpha, b, kfun, kparam):
    """Multiclass kernel SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtrain : ndarray, shape (t, n)
         features used during training (one row per feature vector).
    alpha : ndarray, shape (t, k * (k - 1) // 2)
         matrix of learned coefficients.
    b : ndarray, shape (k * (k - 1) // 2,)
         vector of biases.
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector) in the range 0...(k-1).
    ndarray, shape (m, k)
        classification scores.
    """
    _check_size("mn, tn, ts, s", X, Xtrain, alpha, b)
    # 1) recover the number of classes from s = 1 + 2 + ... + k
    m = X.shape[0]
    s = b.size
    k = int(1 + np.sqrt(1 + 8 * s)) // 2
    votes = np.zeros((m, k))
    K = kernel(X, Xtrain, kfun, kparam)
    logits = K @ alpha + b
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


def one_vs_one_ksvm_train(X, Y, kfun, kparam, lambda_, lr=1e-3, steps=1000,
                          init_alpha=None, init_b=None):
    """Train a multi-class kernel SVM using the one vs. one strategy.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels.
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_alpha : ndarray, shape (m, k * (k - 1) // 2)
        initial coefficient (None for zero initialization)
    init_b : ndarray, shape (k * (k - 1) // 2,)
        initial biases (None for zero initialization)

    Returns
    -------
    alpha : ndarray, shape (m, k * (k - 1) // 2)
        matrix of learned coefficients.
    b : ndarray(k * (k - 1) // 2,)
        learned biases.
    """
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    alpha = np.zeros((m, k * (k - 1) // 2))
    b = np.zeros(k * (k - 1) // 2)
    j = 0
    # For each pair of classes...
    for pos in range(k):
        for neg in range(pos + 1, k):
            # Build a training subset
            subset = (np.logical_or(Y == pos, Y == neg)).nonzero()[0]
            Xbin = X[subset, :]
            Ybin = (Y[subset] == pos)
            a1 = (None if init_alpha is None else init_alpha[subset, j])
            b1 = (0 if init_b is None else init_b[j])
            # Train the classifier
            abin, bbin = ksvm_train(Xbin, Ybin, kfun, kparam, lambda_, lr=lr,
                                    steps=steps, init_alpha=a1, init_b=b1)
            alpha[subset, j] = abin
            b[j] = bbin
            j += 1
    return alpha, b


def one_vs_rest_ksvm_inference(X, Xtrain, alpha, b, kfun, kparam):
    """Multiclass kernel SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtrain : ndarray, shape (t, n)
         features used during training (one row per feature vector).
    alpha : ndarray, shape (t, k)
         matrix of learned coefficients.
    b : ndarray, shape (k,)
         vector of biases.
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector) in the range 0...(k-1).
    ndarray, shape (m, k)
        classification scores.
    """
    _check_size("mn, tn, tk, k", X, Xtrain, alpha, b)
    K = kernel(X, Xtrain, kfun, kparam)
    logits = K @ alpha + b
    labels = np.argmax(logits, 1)
    return labels, logits


def one_vs_rest_ksvm_train(X, Y, kfun, kparam, lambda_, lr=1e-3, steps=1000,
                           init_alpha=None, init_b=None):
    """Train a multi-class kernel SVM using the one vs. rest strategy.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels.
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_alpha : ndarray, shape (m, k * (k - 1) // 2)
        initial coefficient (None for zero initialization)
    init_b : ndarray, shape (k * (k - 1) // 2,)
        initial biases (None for zero initialization)

    Returns
    -------
    alpha : ndarray, shape (m, k * (k - 1) // 2)
        matrix of learned coefficients.
    b : ndarray(k * (k - 1) // 2,)
        learned biases.
    """
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    alpha = np.zeros((m, k))
    b = np.zeros(k)
    for c in range(k):
        Ybin = (Y == c)
        a1 = (None if init_alpha is None else init_alpha[:, c])
        b1 = (0 if init_b is None else init_b[c])
        abin, bbin = ksvm_train(X, Ybin, kfun, kparam, lambda_, lr=lr,
                                steps=steps, init_alpha=a1, init_b=b1)
        alpha[:, c] = abin
        b[c] = bbin
    return alpha, b
