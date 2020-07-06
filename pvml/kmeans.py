import numpy as np
from .gda import mindist_inference
from .checks import _check_size


def kmeans_train(X, k, steps=1000, init_centroids=None):
    """K-means clustering algorithm.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    k : int
        number of clusters.
    steps : int
        maximum number of iterations.
    init_centroids : ndarray, shape (k, n)
        initial centroid (None for random initialization).

    Returns
    -------
    centroids : ndarray, shape (k, n)
         class centroids.
    """
    _check_size("mn, kn?", X, init_centroids)
    _check_centroids(X, k, init_centroids)
    m, n = X.shape
    # Initialization
    if init_centroids is None:
        centroids = np.empty((k, n))
        Y = np.arange(m) % k
        np.random.shuffle(Y)
    else:
        centroids = init_centroids
        Y, _ = kmeans_inference(X, centroids)
    # Main loop
    for step in range(steps):
        # Update the centroids
        counts = np.bincount(Y, minlength=k)
        while (counts == 0).any():
            _respawn_empty(Y, counts)
        for i in range(k):
            centroids[i, :] = X[Y == i, :].mean(0)
        # Update the labels
        Yold = Y
        Y, _ = kmeans_inference(X, centroids)
        # Stop if nothing changed
        if (Y == Yold).all():
            break
    return centroids


def kmeans_inference(X, centroids):
    """K-means inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    centroids : ndarray, shape (k, n)
         class mean vectors.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, k)
        scores assigned to each class.
    """
    # After training K-means is just a minimum distance classifier
    return mindist_inference(X, centroids)


def _respawn_empty(Y, counts):
    # Divide the samples in the largest cluster to replace the first empty one.
    big = np.argmax(counts)
    empty = (counts == 0).nonzero()[0][0]
    Ynew = np.random.choice(np.array([empty, big]), (counts[big],))
    changes = (Ynew == empty).sum()
    Y[Y == big] = Ynew
    counts[empty] = changes
    counts[big] -= changes


def _check_centroids(X, k, centroids):
    if k < 1:
        msg = "The number of clusters ({}) must be a strictly positive integer"
        raise ValueError(msg.format(k))
    if centroids is not None and k != centroids.shape[0]:
        msg = "The number of clusters ({}) "
        msg += "does not match the number of centroids ({})"
        raise ValueError(msg.format(k, centroids.shape[0]))
    if k > X.shape[0]:
        msg = "The number of samples ({}) "
        msg += "is smaller than the number of clusters ({})"
        raise ValueError(msg.format(X.shape[0], k))
