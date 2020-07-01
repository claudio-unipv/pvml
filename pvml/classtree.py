import numpy as np
# from .checks import _check_classification
from checks import _check_classification
from utils import one_hot_vectors

# TODO:
# - docstrings & comments
# - CC Pruning
# - categorical variables and splits


class ClassificationTree:
    def __init__(self):
        self._reset(1, 1)

    def inference(self, X):
        m = X.shape[0]
        J = np.zeros(m, dtype=int)
        while True:
            s = (~self.terminal[J]).nonzero()[0]
            if s.size == 0:
                break
            c = (X[s, self.feature[J[s]]] < self.threshold[J[s]])
            J[s] = self.children[J[s], 1 - c]  # 0 -> left, 1 -> right
        probs = self.distribution[J, :]
        labels = probs.argmax(1)
        return labels, probs

    def train(self, X, Y, diversity="gini", minsize=10):
        Y = _check_classification(X, Y)
        dfun = _DIVERSITY_FUN[diversity]
        m, n = X.shape
        k = Y.max() + 1
        maxleaves = 2 * m // minsize
        self._reset(2 * maxleaves - 1, k)
        H = one_hot_vectors(Y, k)
        self._grow(X, H, 0, dfun, minsize)
        self._trim()

    def _reset(self, nodes, classes):
        self.children = np.zeros((nodes, 2), dtype=int)
        self.feature = np.zeros(nodes, dtype=int)
        self.terminal = np.ones(nodes, dtype=np.bool)
        self.threshold = np.zeros(nodes)
        self.distribution = np.ones((nodes, classes))
        self.nodes = 1

    def _trim(self):
        n = self.nodes
        self.children = self.children[:n, :]
        self.feature = self.feature[:n]
        self.terminal = self.terminal[:n]
        self.threshold = self.threshold[:n]
        self.distribution = self.distribution[:n, :]

    def _grow(self, X, H, t, dfun, minsize):
        m, k = H.shape
        self.distribution[t, :] = (H.sum(0) + 1) / (m + k)
        split = _find_split(X, H, dfun, minsize)
        if split is None:
            return
        self.feature[t] = split[0]
        self.threshold[t] = split[1]
        self.terminal[t] = 0
        self.children[t, 0] = self.nodes
        self.children[t, 1] = self.nodes + 1
        self.nodes += 2
        J = (X[:, split[0]] < split[1])
        self._grow(X[J, :], H[J, :], self.children[t, 0], dfun, minsize)
        self._grow(X[~J, :], H[~J, :], self.children[t, 1], dfun, minsize)

    def _dump(self, indent=0, node=0):
        print("{:3d}{} ".format(node, " " * indent), end="")
        if self.terminal[node]:
            print("==> {}".format(self.distribution[node, :].argmax()))
        else:
            f = self.feature[node]
            t = self.threshold[node]
            print("if x[{}] < {}:".format(f, t))
            self._dump(indent + 4, self.children[node, 0])
            print("   {} ".format(" " * indent), end="")
            print("else:")
            self._dump(indent + 4, self.children[node, 1])


def _find_split(X, H, criterion, minsize):
    """Helper function that finds the best split.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    H : ndarray, shape (m, k)
         one hot vectors for class labels.
    criterion : function
         function mapping probabilities to the real-values objective.
    minsize : int
         minimum size of the subsets in which X is divided by the split.

    Returns
    -------
    int
        feature used to split the data.
    float
        split threshold.
    float
        value of the criterion function for the split.

    The function returns None if no split is able to improve the
    criterion on the whole set.

    """
    m, n = X.shape
    if m < 2 * minsize:
        return None
    p = (1 + H.sum(0, keepdims=True)) / (m + H.shape[1])
    start_criterion = criterion(p)
    split = None
    for j in range(n):
        ret = _decision_stump(X[:, j], H, criterion, minsize)
        if ret is not None and (split is None or ret[1] < split[2]):
            split = j, ret[0], ret[1]
    if split is not None and split[2] < start_criterion:
        return split
    else:
        return None


_DIVERSITY_FUN = {
    "gini": lambda p: (1 - (p ** 2).sum(1)),
    "entropy": lambda p: (-(p * np.nan_to_num(np.log(p))).sum(1)),
    "error": lambda p:  (1 - p.max(1))
}


def _decision_stump(x, h, criterion, minsize):
    """Helper function that finds the best split on a single feature.

    Parameters
    ----------
    x : ndarray, shape (m,)
         input values.
    h : ndarray, shape (m, k)
         one hot vectors for class labels.
    criterion : function
         function mapping probabilities to the real-values objective.
    minsize : int
         minimum size of the subsets in which X is divided by the split.

    Returns
    -------
    float
        split threshold.
    float
        value of the criterion function for the split.

    The function returns None if no split is possible.
    """
    m, k = h.shape
    # 1) sort the data and find candidate splits
    ii = np.argsort(x)
    # possible split points are those where consecutive values are
    # different, except for the first and last minsize positions.
    sp = (x[ii[minsize - 1:m - minsize]] < x[ii[minsize:m - minsize + 1]])
    sp = sp.nonzero()[0] + minsize - 1
    if sp.size == 0:
        return None
    # 2) compute the class distributions at the split points
    counts_lo = h[ii, :].cumsum(0)
    counts_hi = counts_lo[-1:, :] - counts_lo
    p_lo = ((1 + counts_lo[sp, :]) /
            (k + counts_lo[sp, :].sum(1, keepdims=True)))
    p_hi = ((1 + counts_hi[sp, :]) /
            (k + counts_hi[sp, :].sum(1, keepdims=True)))
    p_split = sp / m
    # 3) compute the average criterion and select the lowest
    cost = p_split * criterion(p_lo) + (1 - p_split) * criterion(p_hi)
    j = cost.argmin()
    # 4) return the split value (midway the two consecutvie samples)
    # and the corresponding cost
    best = sp[j]
    threshold = (x[ii[best + 1]] + x[ii[best]]) / 2
    return (threshold, cost[j])


tree = ClassificationTree()
X, Y = np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-2, 2, 4))
X = np.stack([X.ravel(), Y.ravel()], 1)
Y = ((X[:, 0] > 0) + 2 * (X[:, 1] > 0))
X = np.random.randn(1000, 10)
Y = np.random.randint(0, 5, (1000,), dtype=int)
tree.train(X, Y, minsize=1)
print("OK")
Yhat, P = tree.inference(X)
print((Y == Yhat).mean())
print(tree.distribution)
tree._dump()
print(Y)
print(Yhat)
