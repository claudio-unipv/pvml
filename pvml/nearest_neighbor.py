import numpy as np
from .utils import squared_distance_matrix
from .checks import _check_size, _check_labels


def knn_inference(X, Xtrain, Ytrain, k=1):
    """K-Nearest Neighbors prediction of class labels.

    The function also estimates the posterior probabilities as
    fraction of neighbors voting for each class.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtrain : ndarray, shape (t, n)
         reference (training) features (one row per feature vector).
    Ytrain : ndarray, shape (t,)
         reference (training) labels (integers in the range 0...(c - 1) ).
    k : int
         number of neighbors.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m, c)
        probability estimates (one per feature vector).

    """
    X = np.asarray(X)
    Xtrain = np.asarray(Xtrain)
    Ytrain = np.asarray(Ytrain).astype(int)
    _check_size("mn, tn, t", X, Xtrain, Ytrain)
    Ytrain = _check_labels(Ytrain)
    m = X.shape[0]
    classes = Ytrain.max() + 1
    D = squared_distance_matrix(X, Xtrain)
    if k == 1:
        index = np.argmin(D, 1)
        labels = Ytrain[index]
        probs = np.zeros((m, classes))
        probs[np.arange(m), labels] = 1
    else:
        neighs = np.argpartition(D, k, 1)[:, :k]
        counts = _bincount_rows(Ytrain[neighs], classes)
        labels = np.argmax(counts, 1)
        probs = counts / k
    return labels, probs


def knn_select_k(X, Y, maxk=101):
    """Leave-one-out selection of the number of neighbors for KNN.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         reference (training) features (one row per feature vector).
    Y : ndarray, shape (m,)
         reference (training) labels (integers in the range 0...(c - 1) ).
    maxk : int
         maximum value of k that is going to be evaluated.

    Returns
    -------
    int
        best value found for k.
    float
        accuracy estimated for the best k.
    """
    X = np.asarray(X)
    Y = np.asarray(Y).astype(int)
    _check_size("mn, m", X, Y)
    Y = _check_labels(Y)
    D = squared_distance_matrix(X, X)
    m = X.shape[0]
    classes = Y.max() + 1
    np.fill_diagonal(D, np.inf)
    neighs = np.argsort(D, 1)
    best_k = 1
    best_acc = -1
    for k in range(1, min(m, maxk + 1), 2):
        counts = _bincount_rows(Y[neighs[:, :k]], classes)
        labels = np.argmax(counts, 1)
        accuracy = (labels == Y).mean()
        if accuracy > best_acc:
            best_acc = accuracy
            best_k = k
    return best_k, best_acc


def _bincount_rows(X, values):
    """Compute one histogram per row.

    np.bincount works only on 1D arrays.  This extends it to work on
    each row.
    """
    m = X.shape[0]
    idx = X.astype(int) + values * np.arange(m)[:, np.newaxis]
    c = np.bincount(idx.ravel(), minlength=values * m)
    return c.reshape(-1, values)
