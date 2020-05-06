import numpy as np
from .mlp import relu, softmax


class CNN:
    """Convolutional neural network.

    It is composed by a sequence of convolutional layers, followed by
    an average pooling and a softmax.

    The activation function is relu (except for the last convolution).

    The loss function used during training is the cross entropy.

    The network processes four dimensional arrays representing batch
    of images.  The four dimensions represents (1) batch size (2)
    height (3) width (4) channels.

    """

    def __init__(self, channels, kernel_sz=None, strides=None):
        """Create and initialiaze the CNN.

        Parameters
        ----------
        channels : list
            number of channels in the layers (first input, then hiddens,
            then output).  At least two are required.
        kernel_sz : list
            size (height and width) of the spatial filters used by
            convolutions.  Must contain one size for each layer,
            except input.  If None, all filters are 3x3.

        strides : list
            stride (vertical and horizontal) used to apply filters.
            Must contain one for each layer, except input.  If None
            all strides are one.

        For instance, CNN([3, 16, 10], [5, 3], [2, 1]) creates a CNN
        with two convolutional layers: the first is a 5x5 convolution
        with stride 2, 3 input and 16 output channels.  The second is
        a 3x3 convolution with stride 1, 16 input and 10 output
        channels.

        """
        # Defaults
        kernel_sz = (kernel_sz or [3] * (len(channels) - 1))
        self.strides = (strides or [1] * (len(channels) - 1))
        # Initialize weights with the Kaiming technique
        self.weights = [
            np.random.randn(k, k, m, n) * np.sqrt(2.0 / (k * k * m))
            for m, n, k in zip(channels[:-1], channels[1:], kernel_sz)
        ]
        # Biases are zero-initialized
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
        terminated by the activations in the output layer.

        """
        activations = [X]
        for W, b, s in zip(self.weights, self.biases, self.strides):
            X = _convolution(X, W, s, s) + b
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
            the list of computed derivatives, one for each layer.
        """
        d = activations[-1].copy()
        d[np.arange(Y.shape[0]), Y] -= 1  # Subtract the one-hot vectors
        sz = activations[-2].shape[1] * activations[-2].shape[2]
        d /= (sz * activations[-1].shape[0])
        d = d[:, None, None, :].repeat(activations[-2].shape[1], 1)
        d = d.repeat(activations[-2].shape[2], 2)
        derivatives = [d]
        for W, X, s in zip(self.weights[:0:-1], activations[-3::-1],
                           self.strides[:0:-1]):
            d = _convolution_backprop(d, W, X.shape[1], X.shape[2], s, s)
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
        X : ndarray, shape (m, h, w, n)
            input images (m images of n channels).
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
        for X, D, W, b, s, uw, ub in zip(activations, derivatives,
                                         self.weights, self.biases,
                                         self.strides,
                                         self.update_w, self.update_b):
            grad_b = D.sum(2).sum(1).sum(0)
            grad_W = _convolution_derivative(X, D, W.shape[0],
                                             W.shape[1], s, s)
            grad_W += lambda_ * W
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
        X : ndarray, shape (m, h, w, n)
            input images (m images of n channels).

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
        X : ndarray, shape (m, h, w, n)
            input images (m images of n channels).
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
            self.backpropagation(X[indices[i:i + batch], :, :, :],
                                 Y[indices[i:i + batch]], lr=lr,
                                 lambda_=lambda_, momentum=momentum)
            i += batch

    def save(self, filename):
        """Save the network to the file."""
        np.savez(filename, weights=self.weights, biases=self.biases,
                 strides=self.strides)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        data = np.load(filename)
        neurons = [w.shape[0] for w in data["weights"]]
        neurons.append(data["weights"][-1].shape[1])
        network = cls(neurons)
        network.weights = data["weights"]
        network.biases = data["biases"]
        network.strides = data["strides"]
        return network


# The convolution operator and its derivatives are implemented as
# described in the blog post:
#
# https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710


def _im2col(X, kh, kw, sh, sw):
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
        Array of neighborhoods (nh = (h - kh + sh) // sh,
        nw = (w - kw + sw) // sw, nn = n * kh * kw).

    """
    m, h, w, n = X.shape
    shape = (m, (h - kh + sh) // sh, (w - kw + sw) // sw, kh, kw, n)
    tm, th, tw, tn = X.strides
    strides = (tm, sh * th, sw * tw, th, tw, tn)
    Y = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides,
                                        writeable=False)
    return Y.reshape(m, shape[1], shape[2], kh * kw * n)


def _convolution(X, W, sh, sw):
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
        Result of the convolution (nh = (h - kh + sh) // sh,
        nw = (w - kw + sw) // sw).

    Output[b, i, j, c] =
       sum[u,v,d]( X[b, sh * i + u, sw * j + v, d] * W[u, v, d, c] )
    """
    assert X.ndim == 4
    assert X.shape[3] == W.shape[2]
    assert X.shape[1] >= W.shape[0]
    assert X.shape[2] >= W.shape[1]
    Y = _im2col(X, W.shape[0], W.shape[1], sh, sw)
    Z = Y.reshape(-1, Y.shape[3]) @ W.reshape(-1, W.shape[3])
    return Z.reshape(Y.shape[0], Y.shape[1], Y.shape[2], -1)


def _dilate(X, h, w, ph, pw, sh, sw):
    """Dilation operator.

    Insert gaps (filled with zeros) between the elments of X.

    Parameters
    ----------
    X : ndarray, shape (m, h, w, n)
        input images (m images of n channels).
    h : int
        height of the result.
    w : int
        width of the result.
    ph : int
        row where the top elements of X are placed.
    pw : int
        column where the leftmost elements of X are placed.
    sh : int
        vertical stride.
    sw : int
        horizontal stride.

    Returns
    -------
    ndarray, shape (m, h, w, n)

    """
    #          00000
    # 12       01020
    # 34  == > 00000
    #          03040
    #          00000
    b, hin, win, c = X.shape
    Y = np.zeros((b, h, w, c))
    Y[:, ph:(ph + sh * hin):sh, pw:(pw + sw * win):sw, :] = X
    return Y


def _convolution_backprop(D, W, h, w, sh, sw):
    """Return the derivative of the convolution.

    Parameters
    ----------
    D : ndarray, shape (m, hd, wd, c)
        derivative of the result of the convolution
    W : ndarray, shape (kh, kw, n, c)
        filter coefficients.
    h : int
        height of the argument of the convolution
    w : int
        width of the argument of the convolution
    sh : int
        vertical stride.
    sw : int
        horizontal stride.

    Returns
    -------
    ndarray, shape (m, h, w, n)
        derivative of the convolution with respect to its argument.

    h and w are required, since the size of the argument cannot be
    inferred when the stride is greater than one.
    """
    # Dilate and pad the derivative
    kh = W.shape[0]
    kw = W.shape[1]
    D = _dilate(D, h + kh - 1, w + kw - 1, kh - 1, kw - 1, sh, sw)
    # The coefficient of the filters are mirrored and transposed
    W1 = W[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
    return _convolution(D, W1, 1, 1)


def _convolution_derivative(X, D, kh, kw, sh, sw):
    """Derivative with respect to the filter coefficients.

    Parameters
    ----------
    X : ndarray, shape (m, h, w, n)
        input images (m images of n channels).
    D : ndarray, shape (m, hd, wd, c)
        derivative of the result of the convolution
    kh : int
        height of the convolutional kernel
    kw : int
        width of the convolutional kernel
    sh : int
        vertical stride
    sw : int
        horizontal stride

    Returns
    -------
    ndarray, shape (kh, kw, n, c)
        derivatives with respect to the filter coefficients
    """
    h = 1 + sh * (D.shape[1] - 1)
    w = 1 + sw * (D.shape[2] - 1)
    D = _dilate(D, h, w, 0, 0, sh, sw)
    D = D.transpose(1, 2, 0, 3)
    X = X.transpose(3, 1, 2, 0)
    X = X[:, :h + kh - 1, :w + kw - 1, :]
    R = _convolution(X, D, 1, 1)
    return R.transpose(1, 2, 0, 3)
