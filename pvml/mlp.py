import numpy as np
from multinomial_logistic import softmax


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
            the list of computed derivatives, starting from output
            activations and back to the first hidden layer.
        """
        delta = activations[-1].copy()
        delta[np.arange(Y.shape[0]), Y] -= 1  # Subtract the one-hot vectors
        deltas = [delta]
        for W, X in zip(self.weights[:0:-1], activations[-2::-1]):
            delta = delta @ W.T
            deltas.append(delta)
            delta *= (X > 0).astype(int)  # derative of relu
        return deltas[::-1]

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

        Returns
        -------
        float
            the average loss obtained in the forward step..

        """
        # Forward pass
        activations = self.forward(X)
        # Compute the average cross entropy loss
        probs = activations[-1][np.arange(Y.shape[0]), Y]
        loss = -np.log(probs).mean()
        # Backward pass
        deltas = self.backward(Y, activations)
        # Update the parameters
        for X, D, W, b, uw, ub in zip(activations, deltas,
                                      self.weights, self.biases,
                                      self.update_w, self.update_b):
            grad_W = (X.T @ D) / X.shape[0] + 0.5 * lambda_ * W
            grad_b = D.mean(0)
            uw *= momentum
            uw -= lr * grad_W
            W += uw
            ub *= momentum
            ub -= lr * grad_b
            b += ub
        return loss

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

        Applu multiple steps of stochastic gradient descent.

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

        Returns
        -------
        list
            the list of loss values obtained during training.

        """
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
            loss = self.backpropagation(X[indices[i:i + batch], :],
                                        Y[indices[i:i + batch]],
                                        lr=lr,
                                        lambda_=lambda_,
                                        momentum=momentum)
            losses.append(loss)
            i += batch
        return losses


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


if __name__ == "__main__":
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
