import numpy as np


# TODO:
# - docstrings & comments
# - CC Pruning
# - categorical variables and splits


class Node:
    def inference(self, X, classes):
        pass

    def _dump(self, indent):
        pass


class TerminalNode(Node):
    def __init__(self, distribution):
        self.distribution = distribution

    def inference(self, X, classes):
        return np.tile(self.distribution, (X.shape[0], 1))

    def _dump(self, indent):
        s = np.array_str(self.distribution, precision=4)
        p = self.distribution.argmax()
        return (" " * indent) + s + " => " + str(p) + "\n"


class InternalNode(Node):
    def __init__(self, left, right, feature, value):
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value

    def inference(self, X, classes):
        out = np.empty((X.shape[0], classes))
        ii = (X[:, self.feature] < self.value)
        if ii.any():
            out[ii, :] = self.left.inference(X[ii, :], classes)
        if not ii.all():
            ii = np.logical_not(ii)
            out[ii, :] = self.right.inference(X[ii, :], classes)
        return out

    def _dump(self, indent):
        i = (" " * indent)
        hd = i + "if x[{}] < {}:\n".format(self.feature, self.value)
        lb = self.left._dump(indent + 4)
        el = i + "else:\n"
        rb = self.right._dump(indent + 4)
        return "".join([hd, lb, el, rb])


class ClassificationTree:
    def __init__(self):
        self.root = TerminalNode(0)
        self.classes = 1

    def inference(self, X):
        probs = self.root.inference(X, self.classes)
        labels = probs.argmax(1)
        return labels, probs

    def train(self, X, Y, diversity="gini", minsize=10, classes=None):
        dfun = _DIVERSITY_FUN[diversity]
        if classes is None:
            self.classes = Y.max() + 1
        m = X.shape[0]
        H = np.zeros((m, self.classes), dtype=int)
        H[np.arange(m), Y] = 1

        def _train_rec(X, H):
            if X.shape[0] < minsize:
                dist = (1 + H.sum(0)) / (X.shape[0] + self.classes)
                return TerminalNode(dist)
            split = _find_split(X, H, dfun, minsize)
            if split is None:
                dist = (1 + H.sum(0)) / (X.shape[0] + self.classes)
                return TerminalNode(dist)
            ii = (X[:, split[0]] < split[1])
            left = _train_rec(X[ii, :], H[ii, :])
            ii = np.logical_not(ii)
            right = _train_rec(X[ii, :], H[ii, :])
            return InternalNode(left, right, split[0], split[1])
        self.root = _train_rec(X, H)


def _find_split(X, H, criterion, minsize):
    m, n = X.shape
    if m < 2 * minsize:
        return None
    split = None
    for j in range(n):
        ret = _decision_stump(X[:, j], H, criterion, minsize)
        if ret is not None and (split is None or ret[1] < split[2]):
            split = j, ret[0], ret[1]
    return split


_DIVERSITY_FUN = {
    "gini": lambda p: (1 - (p ** 2).sum(1)),
    "entropy": lambda p: (-(p * np.nan_to_num(np.log(p))).sum(1)),
    "error": lambda p:  (1 - p.max(1))
}


def _decision_stump(x, h, criterion, minsize):
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
