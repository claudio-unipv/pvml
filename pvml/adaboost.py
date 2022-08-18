import numpy as np
from .checks import _check_size, _check_labels


class AdaBoost:
    """AdaBoost binary classifier.

    Ensemble of decision stumps trained with the AdaBoost algorithm.

    """
    def __init__(self):
        """Create an ensamble without classifiers."""
        self.indices = np.empty(0, dtype=int)
        self.thresholds = np.empty(0)
        self.alphas = np.empty(0)

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
        ndarray, shape (m,)
            scores (positive for class 1, negative for class 0)
        """
        X = np.asarray(X)
        _check_size("mn", X)
        self._check_features(X)
        if self.indices.size == 0:
            scores = np.zeros(X.shape[0])
        else:
            decisions = (X[:, self.indices] >= self.thresholds[None])
            scores = decisions @ (2 * self.alphas)
            scores -= self.alphas.sum()
        labels = (scores >= 0).astype(int)
        return labels, scores

    def train(self, X, Y, iterations, minsize=1):
        """Train the ensemble.

        If the classifier has been already trained, new weak
        classifiers are added to the existing ones.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            training features.
        Y : ndarray, shape (m,)
            binary training labels.
        iterations : int
            number of training iterations. Each one adds one weak classifier to the ensamble.
        minsize : in
            minimum number of training samples separated by each weak classifier.

        """
        X = np.asarray(X)
        Y = np.asarray(Y).astype(int)
        _check_size("mn, m", X, Y)
        Y = _check_labels(Y, 2)
        eps = 1 / (1 + X.shape[0])
        _, scores = self.inference(X)
        w = np.exp(-scores)
        for _ in range(iterations):
            w = w / max(1e-6, w.sum())
            stump = _find_stump(X, w, Y, minsize)
            if stump is None:
                break
            index, threshold, error = stump
            self.indices = np.append(self.indices, index)
            self.thresholds = np.append(self.thresholds, threshold)
            # Since we admit errors larger than 0.5, alpha may result
            # negative, inverting in practice the decision.  This way
            # we don't have to explicitly consider thresholding in two
            # different directions.
            error = min(max(error, eps), 1 - eps)
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas = np.append(self.alphas, alpha)
            outcome = ((X[:, index] >= threshold) == Y)
            w_mult = np.exp(-alpha * (2 * outcome - 1))
            w *= w_mult

    def size(self):
        """Number of weak classifiers."""
        return self.alphas.size

    def _check_features(self, X):
        if self.indices.size > 0 and X.shape[1] < self.indices.max() + 1:
            msg = "Expected feature vectors with at least {} components (got {})."
            raise ValueError(msg.format(self.indices.max() + 1, X.shape[1]))


def _find_stump(x, w, y, minsize):
    """Helper function that finds the best decision stump.

    The decision is:
    - predict label 1 if the selected feature is greater than the threshold;
    - predict label 0 if the selected feature is less than the threshold.

    The error found can be the smallest or the largest.  In the second
    case in the second case the decision must to be inverted.

    Parameters
    ----------
    x : ndarray, shape (m, n)
         input values.
    w : ndarray, shape (m,)
         weights.
    y : ndarray, shape (m,)
         binary class labels.
    minsize : int
         minimum number of elements before and after the threshold.

    Returns
    -------
    int
        selected features.
    float
        threshold value.
    float
        weighted classification error.

    The function returns None if no split is possible.

    """
    best_index = None
    best_threshold = None
    best_error = 0.5
    for i in range(x.shape[1]):
        s = _find_threshold(x[:, i], w, y, minsize)
        # Select the "most extreme" error:
        if s is not None and np.abs(0.5 - s[1]) >= np.abs(0.5 - best_error):
            best_index = i
            best_threshold, best_error = s
    if best_index is None:
        return None
    return (best_index, best_threshold, best_error)


def _find_threshold(x, w, y, minsize):
    """Helper function that finds the best decision threshold for a single feature.

    Since the threshold can be applied in the two directions, a very
    low or a very high error are both good (in the second case the
    decision must to be inverted).

    Parameters
    ----------
    x : ndarray, shape (m,)
         input values.
    w : ndarray, shape (m,)
         weights.
    y : ndarray, shape (m,)
         binary class labels.
    minsize : int
         minimum number of elements before and after the threshold.

    Returns
    -------
    float
        threshold value.
    float
        weighted classification error, computed assuming
        'x >= threshold' as decision rule.

    The function returns None if no split is possible.

    """
    m = x.shape[0]
    if 2 * minsize > m:
        return None
    minsize = max(1, minsize)
    # 1) Sort the data and find candidate splits.
    ii = np.argsort(x)
    # possible split points are those where consecutive values are
    # different, except for the first and last minsize positions.
    sp = (x[ii[minsize - 1:m - minsize]] < x[ii[minsize:m - minsize + 1]])
    sp = sp.nonzero()[0] + minsize - 1
    if sp.size == 0:
        return None
    # 2) Evaluate the split points.
    #
    # CRITERION:
    #
    # The weighted error is linearly related to the margin:
    #
    #   error = (total_weight - margin) / 2
    #
    # Since we want an error that is very high or very low, we
    # maximize its deviation from the half total error:
    #
    #   abs( total_weight / 2 - error ) = abs( margin ) / 2 =
    #
    #   abs( sum{Y[i] * D[i] * W[i]} ) / 2
    #
    # where both the label Y and the decision D are either -1 or +1 (
    # Y[i] = 2 * y[i] - 1 ).  W[i] is the weight of sample i.
    #
    # Omitting the division by 2, and deparating negative decisions
    # (below the threshold) from positive decision (above thresolds)
    # we obtain:
    #
    #   J = abs( sum{Y[i] * W[i] : D[i] > 0} - sum{Y[i] * W[i] : D[i] < 0} ) =
    #
    #   abs( tot - 2 * sum{Y[i] * W[i] : D[i] < 0} )
    #
    # where tot is the same sum of Y[i] * W[i] over all the samples.
    #
    prod = w * (2 * y - 1)
    neg_margins = prod[ii].cumsum()
    tot = neg_margins[-1]
    objective = np.abs(tot - 2 * neg_margins)
    j = objective[sp].argmax()
    # 3) Return the split value (midway the two consecutvie samples)
    # and the corresponding error.
    best = sp[j]
    threshold = (x[ii[best + 1]] + x[ii[best]]) / 2
    error = (w * (y != (x >= threshold))).sum() / w.sum()
    return (threshold, error)
