import numpy as np


def multinomial_logreg_inference(X, W, b):
    """Predict class probabilities.

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
    P : ndarray, shape (m, k)
         probability estimates.
    """
    logits = X @ W + b.T
    return softmax(logits)


def softmax(Z):
    """Softmax operator.

    Parameters
    ----------
    Z : ndarray, shape (m, n)
         input vectors.

    Returns
    -------
    ndarray, shape (m, n)
         data after the softmax has been applied to each row.
    """
    # Subtracting the maximum improves numerical stability
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def one_hot_vectors(Y, classes):
    """Convert an array of labels into a matrix of one-hot vectors.

    Parameters
    ----------
    Y : ndarray, shape (m,)
         labels.
    classes : int
         number of classes.  If None it is deduced from Y.

    Returns
    -------
    ndarray, shape (m, classes)
         One-hot vectors representing the labels Y.
    """
    m = Y.shape[0]
    H = np.zeros((m, classes))
    H[np.arange(m), Y] = 1
    return H


def multinomial_logreg_train(X, Y, lambda_, lr=1e-3, steps=1000,
                             init_w=None, init_b=None):
    """Train a classifier based on multinomial logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with integer values in the range 0...(k-1).
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
    w : ndarray, shape (n, k)
        learned weights (one vector per class).
    b : ndarray, shape (k,)
        vector of biases.
    """
    m, n = X.shape
    k = Y.max() + 1
    W = (init_w if init_w is not None else np.zeros((n, k)))
    b = (init_b if init_b is not None else np.zeros(k))
    H = one_hot_vectors(Y, k)
    for step in range(steps):
        P = multinomial_logreg_inference(X, W, b)
        grad_W = (X.T @ (P - H)) / m + 2 * lambda_ * W
        grad_b = (P - H).mean(0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def cross_entropy(H, P):
    """Average cross entropy.

    Parameters
    ----------
    H : ndarray, shape (m, k)
        one hot vectors for the target labels.
    P : ndarray, shape (m, k)
        probability estimates.

    Returns
    -------
    float
        average cross entropy.
    """
    return -(H * np.log(P)).sum(1).mean()
