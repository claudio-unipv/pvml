import numpy as np
from .multinomial_logistic import softmax


class MLP:
    """Multi-layer perceptron.

    A multi-layer perceptron for classification.

    The activation function for the output layer is the softmax
    operator, while hidden neurons use relu.

    The loss function used during training is the cross entropy.

    """

    def __init__(self, neuron_counts):
        """Create and initialiaze the MLP.

        At least two layers must be specified (input and output).

        Parameters
        ----------
        neuron_counts : list
            number of neurons in the layers (first input, then hiddens,
            then output).
        """
        # Initialize weights with the Kaiming technique.
        self.weights = [np.random.randn(m, n) * np.sqrt(2.0 / m)
                        for m, n in zip(neuron_counts[:-1], neuron_counts[1:])]
        # Biases are zero-initialized.
        self.biases = [np.zeros(m) for m in neuron_counts[1:]]
        # Accumulators for the momentum terms
        self.update_w = [np.zeros_like(w) for w in self.weights]
        self.update_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        """Compute the activations of the neurons.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).

        Returns
        -------
        list
            the list of activations of the neurons

        The returned list contains arrays (one per layer) with the
        activations of the layers.  The first element of the list is
        the input X, followed by the activations of hidden layers and
        trminated by the activations in the output layer.

        """
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
        """Compute the derivatives of the loss wrt the activations.

        Parameters
        ----------
        Y : ndarray, shape (m,)
            target output (integer class labels).
        activations : list
            activations computed by the forward method.

        Returns
        -------
        list
            the list of computed derivatives, one for each layer.
        """
        d = activations[-1].copy()
        d[np.arange(Y.shape[0]), Y] -= 1  # Subtract the one-hot vectors
        d /= Y.shape[0]
        derivatives = [d]
        for W, X in zip(self.weights[:0:-1], activations[-2::-1]):
            d = d @ W.T
            derivatives.append(d)
            d *= (X > 0).astype(int)  # derivative of relu
        return derivatives[::-1]

    def loss(self, Y, P):
        """Compute the average cross-entropy."""
        return -np.log(P[np.arange(Y.shape[0]), Y]).mean()

    def backpropagation(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99):
        """Backpropagation algorithm.

        Perform both the forward and the backward steps and update the
        parameters.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Y : ndarray, shape (m,)
            target output (integer class labels).
        lr : float
            learning rate.
        lambda_ : float
            regularization coefficients.
        momentum : float
            momentum coefficient.

        """
        activations = self.forward(X)
        derivatives = self.backward(Y, activations)
        for X, D, W, b, uw, ub in zip(activations, derivatives,
                                      self.weights, self.biases,
                                      self.update_w, self.update_b):
            grad_W = (X.T @ D) + 0.5 * lambda_ * W
            grad_b = D.sum(0)
            uw *= momentum
            uw -= lr * grad_W
            W += uw
            ub *= momentum
            ub -= lr * grad_b
            b += ub

    def inference(self, X):
        """Compute the predictions of the network.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).

        Returns
        -------
        ndarray, shape (m,)
            predicted labels, in the range 0, ..., k - 1
        ndarray, shape (m, k)
            posterior probability estimates.

        """
        probs = self.forward(X)[-1]
        labels = np.argmax(probs, 1)
        return labels, probs

    def train(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99,
              steps=10000, batch=None):
        """Train the network.

        Apply multiple steps of stochastic gradient descent.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Y : ndarray, shape (m,)
            target output (integer class labels).
        lr : float
            learning rate.
        lambda_ : float
            regularization coefficients.
        momentum : float
            momentum coefficient.
        steps : int
            training iterations.
        batch : int or None
            size of the minibatch used in each step.  When None all
            the data is used in each step.

        """
        m = X.shape[0]
        if batch is None:
            batch = X.shape[0]
        i = m
        indices = np.arange(m)
        for step in range(steps):
            if i + batch > m:
                i = 0
                np.random.shuffle(indices)
            self.backpropagation(X[indices[i:i + batch], :],
                                 Y[indices[i:i + batch]],
                                 lr=lr,
                                 lambda_=lambda_,
                                 momentum=momentum)
            i += batch

    def save(self, filename):
        """Save the network to the file."""
        np.savez(filename, weights=self.weights, biases=self.biases)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        data = np.load(filename)
        neurons = [w.shape[0] for w in data["weights"]]
        neurons.append(data["weights"][-1].shape[1])
        network = cls(neurons)
        network.weights = data["weights"]
        network.biases = data["biases"]
        return network


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def _check_gradient(mlp, eps=1e-6, bsz=2):
    """Numerical check gradient computations."""
    insz = mlp.weights[0].shape[0]
    outsz = mlp.weights[-1].shape[1]
    X = np.random.randn(bsz, insz)
    Y = np.random.randint(0, outsz, (bsz,))
    A = mlp.forward(X)
    D = mlp.backward(Y, A)
    L = mlp.loss(Y, A[-1])
    for XX, W, DD in zip(A, mlp.weights, D):
        grad_W = (XX.T @ DD)
        GW = np.empty_like(W)
        for idx in np.ndindex(*W.shape):
            bak = W[idx]
            W[idx] += eps
            A1 = mlp.forward(X)
            L1 = mlp.loss(Y, A1[-1])
            W[idx] = bak
            GW[idx] = (L1 - L) / eps
        err = np.abs(GW - grad_W).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4
    for b, DD in zip(mlp.biases, D):
        grad_b = DD.sum(0)
        Gb = np.empty_like(b)
        for idx in np.ndindex(*b.shape):
            bak = b[idx]
            b[idx] += eps
            A1 = mlp.forward(X)
            L1 = mlp.loss(Y, A1[-1])
            b[idx] = bak
            Gb[idx] = (L1 - L) / eps
        err = np.abs(Gb - grad_b).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4


if __name__ == "__main__":
    mlp = MLP([3, 7, 2, 10])
    _check_gradient(mlp)
    import sys
    sys.exit()
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            hidden = list(map(int, self.args.hidden.split(",")))
            neurons = [X.shape[1]] + hidden + [Y.max() + 1]
            self.net = MLP(neurons)
            losses = self.net.train(X, Y, steps=self.args.steps,
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    batch=self.args.batch)
            return losses

        def inference(self, X):
            p = self.net.forward(X)[-1]
            Y = np.argmax(p, 1)
            return Y

    app = Demo()
    app.parser.add_argument("-m", "--momentum", type=float,
                            default=0.99, help="Momentum term")
    app.parser.add_argument("-H", "--hidden", default="",
                            help="Comma separated sizes of hidden layers")
    app.parser.add_argument("-B", "--batch", type=int,
                            help="Size of the minibatch")
    app.run()
