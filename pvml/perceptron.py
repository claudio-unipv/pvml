import numpy as np


#!begin1
def perceptron_train(X, Y, steps=10000):
    """Train a binary classifier using the perceptron algorithm.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    steps: int
        maximum number of training iterations

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    loss : ndarray
        errors made in each training iteration.
    """
    w = np.zeros(X.shape[1])
    b = 0
    loss = np.zeros(steps)
    for step in range(steps):
        for i in range(X.shape[0]):
            d = (1 if X[i, :] @ w + b > 0 else 0)
            w += (Y[i] - d) * X[i, :].T
            b += (Y[i] - d)
            loss[step] += np.abs(Y[i] - d)
        if loss[step] == 0:
            loss = loss[:step]
            break
    return w, b, loss
#!end1


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            w, b, losses = perceptron_train(X, Y, steps=self.args.steps)
            self.w = w
            self.b = b
            return losses

        def inference(self, X):
            return (X @ self.w + self.b > 0).astype(int)

    Demo().run()
