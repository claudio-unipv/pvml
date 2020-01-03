import numpy as np


#!begin1
def svm_inference(X, w, b):
    """SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    w : ndarray, shape (n,)
         weight vector.
    b : float
         scalar bias.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m,)
        classification scores (one per feature vector).
    """
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits


def hinge_loss(labels, logits):
    """Average hinge loss.

    Parameters
    ----------
    labels : ndarray, shape (m,)
        binary target labels (0 or 1).
    logits : ndarray, shape (m,)
        classification scores (logits).

    Returns
    -------
    float
        average hinge loss.
    """
    loss = np.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()


def svm_train(X, Y, lambda_, lr=1e-3, steps=1000):
    """Train a binary SVM classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    loss : ndarray, shape (steps,)
        loss value after each training iteration.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    C = (2 * Y) - 1
    loss = np.empty(steps)
    for step in range(steps):
        labels, logits = svm_inference(X, w, b)
        regularization = 0.5 * lambda_ * (w ** 2).sum()
        loss[step] = hinge_loss(Y, logits).mean() + regularization
        hinge_diff = -C * ((C * logits) < 1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, loss
#!end1


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            w, b, loss = svm_train(X, Y, self.args.lambda_,
                                   lr=self.args.lr, steps=self.args.steps)
            self.w = w
            self.b = b
            return loss

        def inference(self, X):
            return svm_inference(X, self.w, self.b)[0]
    Demo().run()
