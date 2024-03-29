import numpy as np
from .utils import log_nowarn, squared_distance_matrix
from .checks import _check_size, _check_labels


def hgda_train(X, Y, priors=None):
    """Train a heteroscedastic GDA classifier.

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
    invcovs : ndarray, shape (k, n, n)
         inverse of the class covariance matrices.
    priors : ndarray, shape (k,)
         class prior probabilities.
    """
    X = np.asarray(X)
    Y = np.asarray(Y).astype(int)
    if priors is not None:
        priors = np.asfarray(priors)
    _check_size("mn, m, k?", X, Y, priors)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    invcovs = np.empty((k, n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
        cov = np.cov(X[Y == c, :].T)
        invcovs[c, :, :] = np.linalg.inv(cov)
    return means, invcovs, priors


def hgda_inference(X, means, invcovs, priors):
    """Heteroscedastic GDA inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    means : ndarray, shape (k, n)
         class mean vectors.
    invcovs : ndarray, shape (k, n, n)
         inverse of the class covariance matrices.
    priors : ndarray, shape (k,)
         class prior probabilities.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        scores assigned to each class.
    """
    X = np.asarray(X)
    means = np.asarray(means)
    invcovs = np.asarray(invcovs)
    priors = np.asarray(priors)
    _check_size("mn, kn, knn, k", X, means, invcovs, priors)
    m, n = X.shape
    k = means.shape[0]
    scores = np.empty((m, k))
    for c in range(k):
        det = np.linalg.det(invcovs[c, :, :])
        diff = X - means[c, :]
        q = ((diff @ invcovs[c, :, :]) * diff).sum(1)
        scores[:, c] = 0.5 * q - 0.5 * np.log(det) - log_nowarn(priors[c])
    labels = np.argmin(scores, 1)
    return labels, -scores


def ogda_train(X, Y, priors=None):
    """Train a homoscedastic GDA classifier.

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
    W : ndarray, shape (n, k)
         weight vectors, each row representing a different class.
    b : ndarray, shape (k,)
         vector of biases.
    """
    X = np.asarray(X)
    Y = np.asarray(Y).astype(int)
    if priors is not None:
        priors = np.asfarray(priors)
    _check_size("mn, m, k?", X, Y, priors)
    Y = _check_labels(Y)
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    cov = np.zeros((n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
        cov += priors[c] * np.cov(X[Y == c, :].T)
    icov = np.linalg.inv(cov)
    W = -(icov @ means.T)
    q = ((means @ icov) * means).sum(1)
    b = 0.5 * q - log_nowarn(priors)
    return W, b


def ogda_inference(X, W, b):
    """Homoscedastic GDA inference.

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
        scores assigned to each class.
    """
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)
    _check_size("mn, nk, k", X, W, b)
    scores = X @ W + b.T
    labels = np.argmin(scores, 1)
    return labels, -scores


def mindist_train(X, Y):
    """Train a minimum distance classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.

    Returns
    -------
    means : ndarray, shape (k, n)
         class mean vectors.
    """
    X = np.asarray(X)
    Y = np.asarray(Y).astype(int)
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    k = Y.max() + 1
    n = X.shape[1]
    means = np.empty((k, n))
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
    return means


def mindist_inference(X, means):
    """Minimum distance classifier inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    means : ndarray, shape (k, n)
         class mean vectors.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        scores assigned to each class.
    """
    X = np.asarray(X)
    means = np.asarray(means)
    _check_size("mn, kn", X, means)
    d = squared_distance_matrix(X, means)
    labels = np.argmin(d, 1)
    return labels, -d
