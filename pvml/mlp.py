import numpy as np
from .multinomial_logistic import softmax, cross_entropy


class MLP:
    """Multi-layer perceptron.

    A multi-layer perceptron for classification.

    The activation function for the output layer is the softmax
    operator, while hidden neurons use relu.  The loss function used
    during training is the cross entropy.

    To use a different architecture it is possible to define a new
    derived class which overrides some of the methods.

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
        self.reset_momentum()

    def reset_momentum(self):
        """Create the accumulators for the momentum terms and fill them with zeros."""
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
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.forward_hidden_layer(X, W, b)
            X = self.forward_hidden_activation(X)
            activations.append(X)
        X = self.forward_output_layer(X, self.weights[-1], self.biases[-1])
        X = self.forward_output_activation(X)
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
        d = self.backward_output_activation(Y, activations[-1])
        derivatives = [d]
        if len(self.weights) > 1:
            d = self.backward_output_layer(self.weights[-1], self.biases[-1], d)
            d = self.backward_hidden_activation(activations[-2], d)
            derivatives.append(d)
        for W, b, X in zip(self.weights[-2:0:-1], self.biases[-2:0:-1],
                           activations[-3::-1]):
            d = self.backward_hidden_layer(W, b, d)
            d = self.backward_hidden_activation(X, d)
            derivatives.append(d)
        return derivatives[::-1]

    def parameters_gradient(self, activations, derivatives):
        """Derivatives of the loss with respect to the parameters.

        Parameters
        ----------
        activations : list
            activations computed by the forward method.
        derivatives : list
            derivatives computed by the backward method.

        Returns
        -------
        gradient_weights : list
            list of derivatives with respect to weights.
        graident_biases : list
            list of derivatives with respect to biases.
        """
        gradient_weights = []
        gradient_biases = []
        for X, D in zip(activations, derivatives):
            gradient_weights.append(X.T @ D)
            gradient_biases.append(D.sum(0))
        return gradient_weights, gradient_biases

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
        grad_weights, grad_biases = self.parameters_gradient(activations, derivatives)
        for W, b, uw, ub, grad_W, grad_b in zip(self.weights, self.biases,
                                                self.update_w, self.update_b,
                                                grad_weights, grad_biases):
            uw *= momentum
            uw -= lr * (grad_W + lambda_ * W)
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
        neurons = [w.shape[0] for w in self.weights]
        neurons.append(self.weights[-1].shape[1])
        np.savez(filename, neurons=neurons, *self.weights, *self.biases)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        data = np.load(filename)
        neurons = data["neurons"]
        layers = len(neurons) - 1
        network = cls(neurons)
        for i in range(layers):
            network.weights[i][...] = data["arr_" + str(i)]
            network.biases[i][...] = data["arr_" + str(i + layers)]
        return network

    # These last methods can be modified by derived classes to change
    # the architecture of the MLP.

    def forward_hidden_layer(self, X, W, b):
        """Forward pass of hidden layers."""
        return X @ W + b

    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return relu(X)

    def forward_output_layer(self, X, W, b):
        """Forward pass of the output layer."""
        return X @ W + b

    def forward_output_activation(self, X):
        """Activation function of the output layer."""
        return softmax(X)

    def backward_hidden_layer(self, W, b, d):
        """Backward pass of hidden layers."""
        return d @ W.T

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        return d * (Y > 0).astype(int)

    def backward_output_layer(self, W, b, d):
        """Backward pass of the ouput layer."""
        return d @ W.T

    def backward_output_activation(self, Y, P):
        """Derivative of the activation function of output layer."""
        d = P.copy()
        # Implicitly subtract the one-hot vectors
        d[np.arange(Y.shape[0]), Y] -= 1
        return d / Y.shape[0]

    def backward_to_input(self, d):
        """Derivative with respect to input.

        This is normally not computed since it is not used in
        backpropagation.

        Parameters
        ----------
        d : ndarray, shape (m, n)
            first derivative computed by the backward method.

        Returns
        -------
        ndarray, shape (m, i)
            derivative with respect to input.

        """
        return d @ self.weights[0].T

    def loss(self, Y, P):
        """Compute the average cross-entropy."""
        return cross_entropy(Y, P)


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)
