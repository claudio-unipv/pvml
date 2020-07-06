import numpy as np
from .checks import _check_size, _check_labels
from .utils import one_hot_vectors, log_nowarn


# Implementation
# ==============
#
# It is similar to CART trees, but without the support for regression,
# categorical variables, and other advanced features.
#
# In training a 'full' tree is grown with a high likelihood of
# overfitting the data.  A diversity criterion is used to split the
# data until it is divided in omogeneous subsets.
#
# Then, cost-complexity pruning is used to prune the tree to the
# 'right' level of complexity.  CC pruning optimize the following
# objective:
#
#   min R(T, lambda) = error(T) + lambda * size(T)
#
# where size(T) is the number of leaves, and error(T) is the training
# error of the tree T.  For lambda = 0 the initial tree is optimal.
# As lambda gorws, eventually it will be better to replace a whole
# subtree with a leaf.  This happen for
#
#   lambda = Delta-error(t) / (size(t) - 1)
#
# where t is a subtree, Delta-error is the increase in training error
# that the pruning of t wuold cause, and size(t) is the number of
# leaves in the subtree.
#
# With the formula above candidate values of lambda are selected.
# K-fold cross-validation is used to pick the one minimizing the
# generalization error.


class ClassificationTree:
    """Classification trees.

    CART-like model of a classification tree for continuous variables.
    """
    def __init__(self):
        """Create a tree with a single node."""
        self._reset(1, 1)

    def inference(self, X):
        """Prediction of the class labels.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).

        Returns
        -------
        ndarray, shape (m,)
            predicted labels (one per feature vector).
        ndarray, shape (m, k)
            class probabilities (one per feature vector).
        """
        _check_size("mn", X)
        self._check_features(X)
        J = self._descend(X)
        probs = self.distribution[J, :]
        labels = probs.argmax(1)
        return labels, probs

    def train(self, X, Y, diversity="gini", minsize=1, pruning_cv=5):
        """Train a classification tree.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            training features.
        Y : ndarray, shape (m,)
            training labels.
        diversity : str
            diversity criterion ('gini', 'entropy' or 'error').
        minsize : int
            minimum number of training samples per node during tree growth.
        pruning_cv : int
            number of cross-validation folds used for cost-complexity
            pruning (0 disable pruning).
        """
        _check_size("mn, m", X, Y)
        Y = _check_labels(Y)
        dfun = _DIVERSITY_FUN[diversity]
        m, n = X.shape
        k = Y.max() + 1
        maxleaves = 2 * m // minsize
        self._reset(2 * maxleaves - 1, k)
        H = one_hot_vectors(Y, k)
        self._grow(X, H, 0, dfun, minsize)
        self._trim()
        if pruning_cv != 0:
            self._prune(X, Y, pruning_cv, diversity, minsize)

    def _reset(self, nodes, classes):
        """Initialize the attributes."""
        self.children = np.zeros((nodes, 2), dtype=int)
        self.feature = np.zeros(nodes, dtype=int)
        self.terminal = np.ones(nodes, dtype=np.bool)
        self.threshold = np.zeros(nodes)
        self.distribution = np.ones((nodes, classes))
        self.nodes = 1

    def _descend(self, X):
        """Return the indexes of the tree nodes in which the samples fall."""
        J = np.zeros(X.shape[0], dtype=int)
        while True:
            s = (~self.terminal[J]).nonzero()[0]
            if s.size == 0:
                break
            c = (X[s, self.feature[J[s]]] < self.threshold[J[s]])
            J[s] = self.children[J[s], 1 - c]  # 0 -> left, 1 -> right
        return J

    def _trim(self):
        """Remove unused nodes."""
        n = self.nodes
        self.children = self.children[:n, :]
        self.feature = self.feature[:n]
        self.terminal = self.terminal[:n]
        self.threshold = self.threshold[:n]
        self.distribution = self.distribution[:n, :]

    def _grow(self, X, H, t, dfun, minsize):
        """Grow the tree by trying to split node t."""
        m, k = H.shape
        self.distribution[t, :] = H.mean(0)
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

    def _prune(self, X, Y, cv, diversity, minsize):
        """Cost-complexity pruning."""
        leaves = self._count_leaves()[~self.terminal]
        errors = self._pruning_errors(X, Y)[~self.terminal] / Y.size
        lambdas = (errors / (leaves - 1))
        lambdas = np.unique(lambdas)
        folds = np.arange(Y.size) % cv
        np.random.shuffle(folds)
        val = self._eval_prune_lambda(X, Y, 0.0, folds, diversity, minsize)
        best = (val, 0.0)
        for lambda_ in lambdas:
            val = self._eval_prune_lambda(X, Y, lambda_, folds, diversity, minsize)
            best = max(best, (val, lambda_))
        self._prune_lambda(X, Y, best[1])

    def _eval_prune_lambda(self, X, Y, lambda_, folds, diversity, minsize):
        """Estimate the accuracy of the tree pruned according to lambda_."""
        Yhat = np.empty_like(Y)
        tree = ClassificationTree()
        for fold in np.unique(folds):
            J = (folds == fold)
            tree.train(X[~J], Y[~J], diversity=diversity, minsize=minsize,
                       pruning_cv=0)
            tree._prune_lambda(X[~J], Y[~J], lambda_)
            Yhat[J] = tree.inference(X[J])[0]
        return (Y == Yhat).mean()

    def _prune_lambda(self, X, Y, lambda_):
        """Prune the tree at the given cost-complexity level."""
        leaves = self._count_leaves()
        errors = self._pruning_errors(X, Y) / Y.size
        lambdas = (errors / np.maximum(1, (leaves - 1)))
        self.terminal |= (lambdas <= lambda_)

    def _count_leaves(self):
        """Return the number of descendent leaves for each node."""
        leaves = np.ones(self.nodes, dtype=int)
        ts = (~self.terminal).nonzero()[0]
        for t in ts[::-1]:
            leaves[t] = leaves[self.children[t, 0]] + leaves[self.children[t, 1]]
        return leaves

    def _pruning_errors(self, X, Y):
        """Return the number of additional errors caused by pruning nodes."""
        klass = self.distribution.argmax(1)
        errors = np.zeros(self.nodes, dtype=int)
        errors[0] = (klass[0] != Y).sum()
        J = np.zeros(X.shape[0], dtype=int)
        while True:
            s = (~self.terminal[J]).nonzero()[0]
            if s.size == 0:
                break
            c = (X[s, self.feature[J[s]]] < self.threshold[J[s]])
            J[s] = self.children[J[s], 1 - c]  # 0 -> left, 1 -> right
            w = (Y[s] != klass[J[s]])
            e = np.bincount(J[s], weights=w, minlength=self.nodes)
            errors += e.astype(int)
        errors2 = errors.copy()
        ts = (~self.terminal).nonzero()[0]
        for t in ts[::-1]:
            errors2[t] = errors2[self.children[t, 0]] + errors2[self.children[t, 1]]
        return errors - errors2

    def _dumps(self, indent=0, node=0):
        """Sting representation of the tree."""
        pre = "{:3d}{} ".format(node, " " * indent)
        if self.terminal[node]:
            klass = self.distribution[node, :].argmax()
            s = pre + "==> {}\n".format(klass)
        else:
            f = self.feature[node]
            t = self.threshold[node]
            s = "".join([pre + "if x[{}] < {}:\n".format(f, t),
                         self._dumps(indent + 4, self.children[node, 0]),
                         "   {} else:\n".format(" " * indent),
                         self._dumps(indent + 4, self.children[node, 1])])
        return s

    def _check_features(self, X):
        if X.shape[1] < self.feature.max() + 1:
            msg = "Expected feature vectors with at least {} components (got {})."
            raise ValueError(msg.format(self.feature.max() + 1, X.shape[1]))


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
    p = H.mean(0, keepdims=True)
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


def _entropy(p):
    """Compute entropy, suppressing warnings and preventing errors."""
    logp = log_nowarn(p)
    logp[p == 0] = 0
    return -(p * logp).sum(1)


_DIVERSITY_FUN = {
    "gini": lambda p: (1 - (p ** 2).sum(1)),
    "entropy": _entropy,
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
    p_lo = ((counts_lo[sp, :]) /
            (counts_lo[sp, :].sum(1, keepdims=True)))
    p_hi = ((counts_hi[sp, :]) /
            (counts_hi[sp, :].sum(1, keepdims=True)))
    p_split = sp / m
    # 3) compute the average criterion and select the lowest
    cost = p_split * criterion(p_lo) + (1 - p_split) * criterion(p_hi)
    j = cost.argmin()
    # 4) return the split value (midway the two consecutvie samples)
    # and the corresponding cost
    best = sp[j]
    threshold = (x[ii[best + 1]] + x[ii[best]]) / 2
    return (threshold, cost[j])
