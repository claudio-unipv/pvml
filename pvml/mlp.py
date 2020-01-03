import numpy as np
from multinomial_logistic import softmax


# TODO:
# - momentum
# - train method with minibatch
# - inference method
# - weight decay
# - mnist example
# - docstrings and comments


class MLP:
    """Multi-layer perceptron."""

    def __init__(self, units):
        self.weights = []
        self.biases = []
        for a, b in zip(units[:-1], units[1:]):
            w = np.random.randn(a, b)
            w *= np.sqrt(2.0 / a)  # Kaiming initialization
            self.weights.append(w)
            self.biases.append(np.zeros(b))

    def forward(self, X):
        activations = [X]
        for W, b in zip(self.weights, self.biases):
            X = X @ W + b.T
            if W is not self.weights[-1]:
                X = relu(X)
            else:
                X = softmax(X)
            activations.append(X)
        return activations

    def backward(self, Y, activations):
        delta = activations[-1].copy()
        delta[np.arange(Y.shape[0]), Y] -= 1  # Subtract the one-hot vectors
        deltas = [delta]
        for W, X in zip(self.weights[:0:-1], activations[-2::-1]):
            delta = delta @ W.T
            deltas.append(delta)
            delta *= (X > 0).astype(int)  # derative of relu
        return deltas[::-1]

    def backprop(self, X, Y, lr=1e-4):
        activations = self.forward(X)
        probs = activations[-1][np.arange(Y.shape[0]), Y]
        loss = -np.log(probs).mean()  # Cross entropy
        deltas = self.backward(Y, activations)
        for X, D, W, b in zip(activations, deltas, self.weights, self.biases):
            grad_W = (X.T @ D) / X.shape[0]
            grad_b = D.mean(0)
            W -= lr * grad_W
            b -= lr * grad_b
        return loss

    def inference(self, X):
        probs = self.forward[-1]
        labels = np.argmax(probs, 1)
        return labels, probs

    def train(self, X, Y, lr=1e-4, steps=10000, batch=None):
        m = X.shape[0]
        if batch is None:
            batch = X.shape[0]
        losses = []
        i = m
        indices = np.arange(m)
        for step in range(steps):
            if i + batch > m:
                i = 0
                np.random.shuffle(indices)
            loss = self.backprop(X[indices[i:i + batch], :],
                                 Y[indices[i:i + batch]],
                                 lr=lr)
            losses.append(loss)
            i += batch
        return losses


def relu(x):
    return np.maximum(x, 0)


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            units = [X.shape[1]] + self.args.hidden + [Y.max() + 1]
            self.net = MLP(units)
            losses = self.net.train(X, Y, steps=self.args.steps,
                                    lr=self.args.lr,
                                    batch=self.args.batch)
            return losses

        def inference(self, X):
            p = self.net.forward(X)[-1]
            Y = np.argmax(p, 1)
            return Y

    app = Demo()
    app.parser.add_argument("-H", "--hidden", type=int, nargs="+",
                            default=[], help="Size of hidden layers")
    app.parser.add_argument("-B", "--batch", type=int,
                            help="Size of the minibatch")
    app.run()
