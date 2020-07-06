import numpy as np
from .utils import log_nowarn
from .checks import _check_size, _check_categorical, _check_labels


def categorical_naive_bayes_train(X, Y, priors=None):
    """Train naive Bayes classifier for categorical data.

    Non integer feature values are truncated.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features with values in {0, ..., q - 1}.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.
    priors : ndarray, shape (k,)
        Prior probabilities for the classes (if None they get
        estimated from Y).

    Returns
    -------
    probs : ndarray, shape (k, n, q)
         class/feature probabilities.
    priors : ndarray, shape (k,)
         class prior probabilities.
    """
    _check_size("mn, m, k?", X, Y, priors)
    Y = _check_labels(Y)
    X = _check_categorical(X)
    m, n = X.shape
    q = X.max() + 1
    k = Y.max() + 1
    probs = np.empty((k, n, q))
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        tot = indices.shape[0] + q  # With Laplacian smoothing
        for j in range(n):
            counts = (1 + np.bincount(X[indices, j], minlength=q))
            probs[c, j, :] = counts / tot
    if priors is None:
        priors = np.bincount(Y) / m
    return (probs, priors)


def categorical_naive_bayes_inference(X, probs, priors):
    """Categorical naive Bayes classifier inference.

    Non integer feature values are truncated.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features with values in {0, ..., q - 1}.
    probs : ndarray, shape (k, n, q)
        class mean vectors.
    priors : ndarray, shape (k,)
        class prior probabilities.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        prediction scores.
    """
    _check_size("mn, knq, k", X, probs, priors)
    X = _check_categorical(X)
    q = probs.shape[2]
    X = np.clip(X.astype(int), 0, q - 1)
    m, n = X.shape
    k = priors.shape[0]
    scores = log_nowarn(priors)[None, :].repeat(m, axis=0)
    for c in range(k):
        for j in range(n):
            scores[:, c] += np.log(probs[c, j, X[:, j]])
    labels = np.argmax(scores, 1)
    return labels, scores


def multinomial_naive_bayes_train(X, Y, priors=None):
    """Train a multinomial naive Bayes classifier.

    Non integer feature values are truncated.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features, they must be non-negative integers.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.
    priors : ndarray, shape (k,)
        Prior probabilities for the classes (if None they get
        estimated from Y).

    Returns
    -------
    W : ndarray, shape (n, k)
         weight matrix.
    b : ndarray, shape (k,)
         vector of biases.
    """
    _check_size("mn, m, k?", X, Y, priors)
    Y = _check_labels(Y)
    X = _check_categorical(X)
    m, n = X.shape
    k = Y.max() + 1
    probs = np.empty((k, n))
    for c in range(k):
        counts = X[Y == c, :].sum(0)
        tot = counts.sum()
        probs[c, :] = (counts + 1) / (tot + n)  # with Laplacian smoothing)
    if priors is None:
        priors = np.bincount(Y) / m
    W = np.log(probs).T
    b = log_nowarn(priors)
    return (W, b)


def multinomial_naive_bayes_inference(X, W, b):
    """Multinomial naive Bayes classifier inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    W : ndarray, shape (n, k)
         weight vectors, each row representing a different class.
    b : ndarray, shape (k,)
         vector of biases.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        prediction scores.
    """
    _check_size("mn, nk, k", X, W, b)
    scores = X @ W + b.T
    labels = np.argmax(scores, 1)
    return labels, scores


def gaussian_naive_bayes_train(X, Y, priors=None):
    """Train a Gaussian Naive Bayes classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.
    priors : ndarray, shape (k,)
        Prior probabilities for the classes (if None they get
        estimated from Y).

    Returns
    -------
    means : ndarray, shape (k, n)
         class mean vectors.
    vars : ndarray, shape (k, n)
         per class variance of features.
    priors : ndarray, shape (k,)
         class prior probabilities.
    """
    _check_size("mn, m, k?", X, Y, priors)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    vars = np.empty((k, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
        vars[c, :] = X[indices, :].var(0)
    return (means, vars, priors)


def gaussian_naive_bayes_inference(X, means, vars, priors):
    """Gaussian Naive Bayes inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    means : ndarray, shape (k, n)
         class mean vectors.
    vars : ndarray, shape (k, n)
         variances.
    priors : ndarray, shape (k,)
         class prior probabilities.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        prediction scores.
    """
    _check_size("mn, kn, k", X, means, priors)
    diffs = (X[:, None, :] - means[None, :, :]) ** 2
    diffs /= vars[None, :, :]
    scores = -0.5 * diffs.sum(2) - 0.5 * np.log(vars).sum(1)[None, :]
    scores += log_nowarn(priors)[None, :]
    labels = np.argmax(scores, 1)
    return labels, scores
