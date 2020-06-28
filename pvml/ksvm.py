import numpy as np
from .checks import _check_classification


def ksvm_inference(X, Xtrain, alpha, b, kfun, kparam):
    """SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtrain : ndarray, shape (t, n)
         features used during training (one row per feature vector).
    alpha : ndarray, shape (t,)
         vector of learned coefficients.
    b : float
         scalar bias.
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m,)
        classification scores (one per feature vector).
    """
    _check_coefficients(Xtrain, alpha, b)
    K = kernel(X, Xtrain, kfun, kparam)
    logits = K @ alpha + b
    labels = (logits > 0).astype(int)
    return labels, logits


def kernel(X1, X2, kfun, kparam):
    """Compute the kernel between two groups of feature vectors.

    Parameters
    ----------
    X1 : ndarray, shape (m, n)
         first group of features (one row per feature vector).
    X2 : ndarray, shape (t, n)
         second group of features (one row per feature vector).
    kfun : string
         name of the kernel function
    kparam : float
         parameter of the kernel

    Returns
    -------
    K : ndarray, shape (m, t)
         matrix with the result of the kernel function applied to
         the two groups of feature vectors.
    """
    _check_kernel(X1, X2)
    if kfun == "polynomial":
        return (X1 @ X2.T + 1) ** kparam
    elif kfun == "rbf":
        qx1 = (X1 ** 2).sum(1, keepdims=True)
        qx2 = (X2 ** 2).sum(1, keepdims=True)
        cross = 2 * X1 @ X2.T
        return np.exp(-kparam * (qx1 - cross + qx2.T))
    raise ValueError("Unknown kernel ('%s')" % kfun)


def ksvm_train(X, Y, kfun, kparam, lambda_, lr=1e-3, steps=1000,
               init_alpha=None, init_b=0):
    """Train a binary non-linear SVM classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
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
    init_alpha : ndarray, shape (m,)
        initial coefficients (None for zero initialization)
    init_b : float
        initial bias

    Returns
    -------
    alpha : ndarray, shape (m,)
        vector of learned coefficients.
    b : float
        learned bias.
    """
    _check_classification(X, Y)
    K = kernel(X, X, kfun, kparam)
    m, n = X.shape
    alpha = (init_alpha if init_alpha is not None else np.zeros(m))
    b = (init_b if init_b is not None else 0)
    C = (2 * Y) - 1
    for step in range(steps):
        ka = K @ alpha
        logits = ka + b
        hinge_diff = -C * ((C * logits) < 1)
        grad_alpha = (hinge_diff @ K) / m + lambda_ * ka
        grad_b = hinge_diff.mean()
        alpha -= lr * grad_alpha
        b -= lr * grad_b
    return alpha, b


def _check_kernel(X1, X2):
    if X1.ndim != 2 or X2.ndim != 2:
        msg = "Features must be a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X1.ndim if X1.ndim != 2 else X2.ndim))
    if X1.shape[1] != X2.shape[1]:
        msg = "Kernel argumens must have the same number of components"
        msg = msg + " (got ({} and {})"
        raise ValueError(msg.format(X1.shape[1], X2.shape[1]))


def _check_coefficients(X, alpha, b):
    if X.ndim != 2:
        msg = "Features must be a bidimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(X.ndim))
    if alpha.ndim != 1:
        msg = "Coefficients must be a one-dimensional array ({} dimension(s) found)"
        raise ValueError(msg.format(alpha.ndim))
    if not np.isscalar(b):
        raise ValueError("The bias must be a scalar")
    if X.shape[0] != alpha.shape[0]:
        msg = "The number of samples ({}) does not match the number of coefficients ({})"
        raise ValueError(msg.format(X.shape[0], alpha.shape[0]))
