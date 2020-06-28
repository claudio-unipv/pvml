import numpy as np


def _check_classification(X, Y):
    """Check that X, Y form a valid training set for classification.

    Also convert Y to int if needed.
    """
    if X.ndim != 2:
        msg = "Features must be a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X.ndim))
    if Y.ndim != 1:
        msg = "Labels must be elements of a vector ({} dimension(s) found)"
        raise ValueError(msg.format(Y.ndim))
    if not np.issubdtype(Y.dtype, np.integer):
        if np.abs(np.modf(Y)[0]).max() > 0:
            raise ValueError("Labels must be integers")
        Y = Y.astype(np.int32)
    if Y.min() < 0:
        raise ValueError("Labels cannot be negative")
    if X.shape[0] != Y.shape[0]:
        msg = "The number of features ({}) does not match the number of labels ({})"
        raise ValueError(msg.format(X.shape[1], Y.shape[0]))
    return Y


def _check_binary_classification(X, Y):
    """Check that X, Y form a valid training set for binary classification."""
    Y = _check_classification(X, Y)
    if Y.max() > 1:
        raise ValueError("Expected binary labels (got {})".format(Y.max()))
    return Y


def _check_categorical(X):
    """Check that X contain categorical data.

    If needed X is converted to int.
    """
    if not np.issubdtype(X.dtype, np.integer):
        if np.abs(np.modf(X)[0]).max() > 0:
            raise ValueError("Categorical data must be integers")
        X = X.astype(np.int32)
    if X.min() < 0:
        raise ValueError("Categorical data cannot be negative")
    return X


def _check_binary_linear(X, w, b):
    """Check that X @ w + b is valid."""
    if X.ndim != 2:
        msg = "Features must form a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X.ndim))
    if w.ndim != 1:
        msg = "Weights must form a one-dimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(w.ndim))
    if not np.isscalar(b):
        raise ValueError("The bias must be a scalar")
    if X.shape[1] != w.shape[0]:
        msg = "The number of features ({}) does not match the number of weights ({})"
        raise ValueError(msg.format(X.shape[1], w.shape[0]))


def _check_linear(X, W, b):
    """Check that X @ W + b is valid."""
    if X.ndim != 2:
        msg = "Features must form a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X.ndim))
    if W.ndim != 2:
        msg = "Weights must form a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(W.ndim))
    if np.isscalar(b) or b.ndim != 1:
        raise ValueError("Bias must form a one-dimensional array")
    if X.shape[1] != W.shape[0]:
        msg = "The number of features ({}) does not match the number of weights ({})"
        raise ValueError(msg.format(X.shape[1], W.shape[0]))
    if W.shape[1] != b.shape[0]:
        msg = "The number of weights ({}) does not match the number of biases ({})"
        raise ValueError(msg.format(W.shape[1], b.shape[0]))


def _check_means(X, M):
    """Check that vectors in X and M have the same number of components."""
    if X.ndim != 2:
        msg = "Features must form a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X.ndim))
    if M.ndim != 2:
        msg = "Means must form a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(M.ndim))
    if X.shape[1] != M.shape[1]:
        msg = "The number of features ({}) does not match the number of weights ({})"
        raise ValueError(msg.format(X.shape[1], M.shape[1]))
