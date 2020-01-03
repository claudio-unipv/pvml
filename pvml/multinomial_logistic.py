import numpy as np


#!begin1
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


def multinomial_logreg_train(X, Y, lambda_, lr=1e-3, steps=1000):
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

    Returns
    -------
    w : ndarray, shape (n, k)
        learned weights (one vector per class).
    b : ndarray, shape (k, )
        vectod of biases.
    loss : ndarray, shape (steps,)
        loss value after each training step.
    """
    m, n = X.shape
    k = Y.max() + 1
    W = np.zeros((n, k))
    b = np.zeros(k)
    H = np.zeros((m, k))  # Matrix of one-hot vectors
    H[np.arange(m), Y] = 1
    loss = np.empty(steps)
    for step in range(steps):
        P = multinomial_logreg_inference(X, W, b)
        loss[step] = cross_entropy(H, P) + lambda_ * (W ** 2).sum()
        grad_W = (X.T @ (P - H)) / m + 2 * lambda_ * W
        grad_b = (P - H).mean(0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b, loss


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
#!end1


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            w, b, losses = multinomial_logreg_train(X, Y,
                                                    self.args.lambda_,
                                                    lr=self.args.lr,
                                                    steps=self.args.steps)
            self.w = w
            self.b = b
            return losses

        def inference(self, X):
            p = multinomial_logreg_inference(X, self.w, self.b)
            Y = np.argmax(p, 1)
            return Y

    Demo().run()
