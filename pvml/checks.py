import numpy as np


def _check_size(spec, *xs):
    """Check that the arrays respect the specification.

    spec is a string of comma separated specifications.  Each
    specification apply to one element in xs.  The specification i a
    list of letters representing the expected number of dimensions (*
    if a scalar is expected).  If the same letter is used multiple
    times, the the dimensions must match.  A traling '?' makes None a
    valid value.

    """
    ss = list(map(str.strip, spec.split(",")))
    if len(ss) != len(xs):
        msg = "Not enough arguments (expected {}, got {})"
        raise ValueError(msg.format(len(ss), len(xs)))
    dims = {}
    for s, x in zip(ss, xs):
        if s.endswith("?"):
            if x is None:
                continue
            s = s[:-1]
        if s == "*":
            if np.isscalar(x):
                continue
            else:
                raise ValueError("Scalar value expected")
        if len(s) != np.ndim(x):
            msg = "Expected an array of {} dimensions ({} dimensions found)"
            raise ValueError(msg.format(len(s), np.ndim(x)))
        for n, d in enumerate(s):
            k = x.shape[n]
            if d not in dims:
                dims[d] = k
            elif k != dims[d]:
                msg = "Dimensions do not agree (got {} and {})"
                raise ValueError(msg.format(dims[d], k))


def _check_labels(Y, nclasses=None):
    """Check that data can represent class labels."""
    if not np.issubdtype(Y.dtype, np.integer):
        if np.abs(np.modf(Y)[0]).max() > 0:
            raise ValueError("Expected integers")
        Y = Y.astype(np.int32)
    if Y.min() < 0:
        raise ValueError("Labels cannot be negative")
    if nclasses is not None and Y.max() >= nclasses:
        msg = "Invalid labels (maximum is {}, got {})"
        raise ValueError(msg.format(nclasses - 1, Y.max()))
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
