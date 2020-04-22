import numpy as np


def meanvar_normalization(X, Xtest=None):
    """Normalize features using moments.

    Linearly normalize each input feature to make it have zero mean
    and unit variance.  Test features, when given, are scaled using
    the statistics computed on X.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True)
    X = X - mu
    X /= np.maximum(sigma, 1e-15)  # 1e-15 avoids division by zero
    if Xtest is None:
        return X
    Xtest = Xtest - mu
    Xtest /= np.maximum(sigma, 1e-15)
    return X, Xtest


def minmax_normalization(X, Xtest=None):
    """Scale features in the [0, 1] range.

    Linearly normalize each input feature in the [0, 1] range.  Test
    features, when given, are scaled using the statistics computed on
    X.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    xmin = X.min(0, keepdims=True)
    xmax = X.max(0, keepdims=True)
    X = X - xmin
    X /= np.maximum(xmax - xmin, 1e-15)  # 1e-15 avoids division by zero
    if Xtest is None:
        return X
    Xtest = Xtest - xmin
    Xtest /= np.maximum(xmax - xmin, 1e-15)
    return X, Xtest


def maxabs_normalization(X, Xtest=None):
    """Scale features in the [-1, 1] range.

    Linearly normalize each input feature in the [-1, 1] range by
    dividing them by the maximum absolute value.  Test features, when
    given, are scaled using the statistics computed on X.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    # 1e-15 avoids division by zero
    amax = np.maximum(np.abs(X).max(0, keepdims=True), 1e-15)
    X = X / amax
    if Xtest is None:
        return X
    Xtest = Xtest / amax
    return X, Xtest


def l2_normalization(X, Xtest=None):
    """L2 normalization of feature vectors.

    Scale feature vectors to make it have unit Euclidean norm.  Test
    features, when given, are scaled as well.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    X = X / np.maximum(q, 1e-15)  # 1e-15 avoids division by zero
    if Xtest is None:
        return X
    q = np.sqrt((Xtest ** 2).sum(1, keepdims=True))
    Xtest = Xtest / np.maximum(q, 1e-15)
    return X, Xtest


def l1_normalization(X, Xtest=None):
    """L1 normalization of feature vectors.

    Scale feature vectors to make it have unit L1 norm.  Test
    features, when given, are scaled as well.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    q = np.abs(X).sum(1, keepdims=True)
    X = X / np.maximum(q, 1e-15)  # 1e-15 avoids division by zero
    if Xtest is None:
        return X
    q = np.abs(Xtest).sum(1, keepdims=True)
    Xtest = Xtest / np.maximum(q, 1e-15)
    return X, Xtest


def whitening(X, Xtest=None):
    """Whitening transform.

    Linearly transform features to make it have zero mean, unit
    variance and null covariance.  Test features, when given, are
    trandformed using the statistics computed on X.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).

    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (returned only when Xtest is not None).

    """
    mu = X.mean(0, keepdims=True)
    sigma = np.cov(X.T)
    evals, evecs = np.linalg.eig(sigma)
    w = (np.maximum(evals, 1e-15) ** -0.5) * evecs  # 1e-15 avoids div. by zero
    X = (X - mu) @ w
    if Xtest is None:
        return X
    Xtest = (Xtest - mu) @ w
    return X, Xtest
