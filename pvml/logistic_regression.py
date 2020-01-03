import numpy as np


#!begin1
def logreg_inference(X, w, b):
    """Predict class probabilities.

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
        probability estimates (one per feature vector).
    """
    logits = X @ w + b
    return 1 / (1 + np.exp(-logits))


def cross_entropy(Y, P):
    """Average cross entropy.

    Parameters
    ----------
    Y : ndarray, shape (m,)
        binary target labels (0 or 1).
    P : ndarray, shape (m,)
        probability estimates.

    Returns
    -------
    float
        average cross entropy.
    """
    eps = 1e-3
    P = np.clip(P, eps, 1 - eps)  # This prevents overflows
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean()
#!end1


#!begin2
def logreg_train(X, Y, lr=1e-3, steps=1000):
    """Train a binary classifier based on logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
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
        loss value after each training step.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss = np.empty(steps)
    for step in range(steps):
        P = logreg_inference(X, w, b)
        loss[step] = cross_entropy(Y, P).mean()
        grad_w = ((P - Y) @ X) / m
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, loss
#!end2


#!begin3
def logreg_l2_train(X, Y, lambda_, lr=1e-3, steps=1000):
    """Train a binary classifier based on L2-regularized logistic regression.

    Parameters
    ----------

    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.

    Returns
    -------

    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    loss : ndarray, shape (steps,)
        loss value after each training step.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss = np.empty(steps)
    for step in range(steps):
        P = logreg_inference(X, w, b)
        loss[step] = cross_entropy(Y, P).mean() + lambda_ * (w ** 2).sum()
        grad_w = ((P - Y) @ X) / m + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, loss
#!end3


#!begin4
def logreg_l1_train(X, Y, lambda_, lr=1e-3, steps=1000):
    """Train a binary classifier based on L1-regularized logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    loss : ndarray, shape (steps,)
        loss value after each training step.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss = np.empty(steps)
    for step in range(steps):
        P = logreg_inference(X, w, b)
        loss[step] = cross_entropy(Y, P).mean() + lambda_ * np.abs(w).sum()
        grad_w = ((P - Y) @ X) / m + lambda_ * np.sign(w)
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, loss
#!end4


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            if self.args.regularization == "l2":
                w, b, losses = logreg_l2_train(X, Y, self.args.lambda_,
                                               lr=self.args.lr,
                                               steps=self.args.steps)
            else:
                w, b, losses = logreg_l1_train(X, Y, self.args.lambda_,
                                               lr=self.args.lr,
                                               steps=self.args.steps)
            self.w = w
            self.b = b
            return losses

        def inference(self, X):
            p = logreg_inference(X, self.w, self.b)
            y = (p > 0.5).astype(int)
            return y
    app = Demo()
    app.parser.add_argument("--regularization", choices=["l2", "l1"],
                            default="l2",
                            help="Regularization function")
    app.run()
