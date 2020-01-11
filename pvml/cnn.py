import numpy as np
from mlp import relu, softmax


# TODO:
# - strides in backward
# - check padding in backward
# - check gradient in general
# - docstrings and comments
# - MNIST example


class CNN:
    """Convolutional neural network.

    A multi-layer perceptron for classification.

    It is composed by a sequence of convolutional layers, followed by
    an average pooling and a softmax.

    The activation function is relu (except for the last convolution).

    The loss function used during training is the cross entropy.

    """

    def __init__(self, channels, kernels=None, strides=None, paddings=None):
        """Create and initialiaze the MLP.

        At least two layers must be specified (input and output).

        Parameters
        ----------
        neuron_counts : list
            number of neurons in the layers (first input, then hiddens,
            then output).
        """
        # Defaults
        kernels = (kernels or [3] * (len(channels) - 1))
        self.strides = (strides or [1] * (len(channels) - 1))
        self.paddings = (paddings or [(k - 1) // 2 for k in kernels])
        # Initialize weights with the Kaiming technique.
        self.weights = [
            np.random.randn(k, k, m, n) * np.sqrt(2.0 / (k * k * m))
            for m, n, k in zip(channels[:-1], channels[1:], kernels)
        ]
        # Biases are zero-initialized.
        self.biases = [np.zeros(m) for m in channels[1:]]
        # Accumulators for the momentum terms
        self.update_w = [np.zeros_like(w) for w in self.weights]
        self.update_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        """Compute the activations of the neurons.

        Parameters
        ----------
        X : ndarray, shape (m, h, w, n)
            input images (m images of n channels).

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
        for W, b, s, p in zip(self.weights, self.biases, self.strides,
                              self.paddings):
            X = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)))
            X = convolution2d(X, W, s, s) + b[None, None, None, :]
            if W is not self.weights[-1]:
                X = relu(X)
            else:
                activations.append(X)
                X = X.mean(1).mean(1)
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
        sz = activations[-2].shape[1] * activations[-2].shape[2]
        delta /= sz
        delta = delta[:, None, None, :].repeat(activations[-2].shape[1], 1)
        delta = delta.repeat(activations[-2].shape[2], 2)
        deltas = [delta]
        for W, X, p in zip(self.weights[:0:-1], activations[-3::-1],
                           self.paddings[:0:-1]):
            delta = np.pad(delta, ((0, 0), (p, p), (p, p), (0, 0)))
            W1 = W[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
            delta = convolution2d(delta, W1, 1, 1)   # !!! strides
            deltas.append(delta)
            delta *= (X > 0).astype(int)  # derivative of relu
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
        for X, D, W, b, uw, ub, p in zip(activations, deltas,
                                         self.weights, self.biases,
                                         self.update_w, self.update_b,
                                         self.paddings):
            grad_b = D.sum(2).sum(1).mean(0)
            D = D.transpose(3, 1, 2, 0)
            X = X.transpose(1, 2, 0, 3)
            D = np.pad(D, ((0, 0), (p, p), (p, p), (0, 0)))  # !!!
            grad_W = convolution2d(D, X, 1, 1)  # !!! stride
            grad_W = grad_W.transpose(1, 2, 3, 0) / X.shape[2]
            grad_W += 0.5 * lambda_ * W
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


def im2col(X, kh, kw, sh, sw):
    """Transform an image into an array of neighborhoods.

    Rectangular neighborhoods of size (kw, kh, n) are linearized into
    a single dimension with size (kw * kh * n).

    Parameters
    ----------
    X : ndarray, shape (m, h, w, n)
        input images (m images of n channels).
    kh : int
        rows in each neighborhood.
    kw : int
        columns in each neighborhood.
    sh : int
        rows between two adjacent neighborhoods (stride).
    sw : int
        columns between two adjacent neighborhoods (stride).

    Returns
    -------
    ndarray, shape (m, nh, nw, nn)
        Array of neighborhoods (nh = (h - kh + 1) // sh,
        nw = (w - kw + 1) // sw, nn = n * kh * kw).

    """
    m, h, w, n = X.shape
    shape = (m, (h - kh + 1) // sh, (w - kw + 1) // sw, kh, kw, n)
    tm, th, tw, tn = X.strides
    strides = (tm, sh * th, sw * tw, th, tw, tn)
    Y = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides,
                                        writeable=False)
    return Y.reshape(m, shape[1], shape[2], kh * kw * n)


def convolution2d(X, W, sh, sw):
    """Two-dimensional convolution of a batch of images.

    Parameters
    ----------
    X : ndarray, shape (m, h, w, n)
        input images (m images of n channels).
    W : ndarray, shape (kh, kw, n, c)
        filter coefficients.
    sh : int
        vertical stride.
    sw : int
        horizontal stride.

    Returns
    -------
    ndarray, shape (m, nh, nw, nn)
        Result of the convolution (nh = (h - kh + 1) // sh,
        nw = (w - kw + 1) // sw).
    """
    Y = im2col(X, W.shape[0], W.shape[1], sh, sw)
    Z = Y.reshape(-1, Y.shape[3]) @ W.reshape(-1, W.shape[3])
    return Z.reshape(Y.shape[0], Y.shape[1], Y.shape[2], -1)


# X = np.arange(49 * 4).reshape(1, 7, 7, 4)
# print(X[0, :, :, 0])
# W = np.zeros((3, 3, 4, 2))
# W[1, 0, 0, 0] = -1
# W[1, 2, 0, 0] = 1
# W[0, 0, 0, 1] = -1


# Z = convolution2d(X, W, 2, 1)
# print(Z[0, :, : ,0])
# print(Z[0, :, : ,1])


# X = np.zeros((7, 16, 16, 3))
# Y = np.arange(7) % 10
# cnn = CNN([3, 20, 10])
# A = cnn.forward(X)
# print("Forward")
# for a in A:
#     print("x".join(map(str, a.shape)))
# D = cnn.backward(Y, A)
# print("Backward")
# for d in D:
#     print("x".join(map(str, d.shape)))
# print("Backprop")
# cnn.backpropagation(X, Y)


"""

H, K, S -> (H - K + 1) // S = O

...............
.1.2.3.4.5.6.7.
...............
.8.9.0.1.2.3.4.
...............

S * (O - 1) + 1 + P

S * (O - 1) + 1 + P - K + 1
S * ( ( (H - K + 1) // S ) - 1 ) + 1 + P - K + 1=
S * ( ( H - K + 1 * S) // S) + 1 + P - K + 1=
H - K + 1 + 1 + P - K + 1 = H
== > P + 2 - 2K = 0
P = 2(K - 1)


"""

def convolution2d_backprop(W, D):
    """Return dJ / dX[...] given D = dJ / d (X * W)[...]"""
    k = 3
    s = 2
    b, h, w, c = D.shape
    h1 = 1 + (h - 1) * s + 2 * (k - 1)
    w1 = 1 + (w - 1) * s + 2 * (k - 1)
    D1 = np.zeros((b, h1, w1, c))
    D1[:, k - 1:s * h + k - 1:s, k - 1:s * w + k - 1:s, :] = D
    W1 = W[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
    return convolution2d(D1, W1, s, s)


def check_gradient(X, W, s=2):
    b, h, w, c = X.shape
    Y = convolution2d(X, W, s, s)
    _, h1, w1, c1 = Y.shape
    e = 1e-5
    for k1 in range(c1):
        for i1 in range(h1):
            for j1 in range(w1):
                DY = np.zeros_like(Y)
                DY[0, i1, j1, k1] = 1
                DX = convolution2d_backprop(W, DY)
                for k in range(c):
                    for i in range(h):
                        for j in range(w):
                            X1 = X.copy()
                            X1[0, i, j, k] += e
                            Y1 = convolution2d(X1, W, s, s)
                            dY = (Y1[0, i1, j1, k1] - Y[0, i1, j1, k1]) / e
                            if np.abs(dY - DX[0, i, j, k]) > 1e-4:
                                print(i, j, k, i1, j1, k1, dY, DX[0, i, j, k])
                            
                

X = np.random.randn(1, 7, 7, 1)
W = np.random.randn(3, 3, 1, 1)
Y = convolution2d(X, W, 1, 1)
print(X.shape, W.shape, "->", Y.shape)
DY = np.ones_like(Y)
DX = convolution2d_backprop(W, DY)
print(DY.shape, "->", DX.shape)

check_gradient(X, W)
