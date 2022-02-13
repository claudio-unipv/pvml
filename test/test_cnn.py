import numpy as np
import pvml
import io
import test_data
import pytest


class CNNTanh(pvml.CNN):
    """CNN with hyperbolic tangent activation function.

    It is used here because tanh is more reliable than ReLU for
    numerical testing.
    """
    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return np.tanh(X)

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        # y = tanh(x)  ==>  dy/dx = (1 - tanh(x)^2) = (1 - y^2)
        return d * (1 - Y ** 2)


def test_activation_gradient():
    """Test the gradient of the activation function."""
    np.random.seed(7477)
    cnn = CNNTanh([1, 1])
    X = np.random.randn(10, 1)
    Y = cnn.forward_hidden_activation(X)
    eps = 1e-7
    Y1 = cnn.forward_hidden_activation(X + eps)
    D = cnn.backward_hidden_activation(Y, np.ones_like(Y))
    D1 = (Y1 - Y) / eps
    error = np.abs(D1 - D).max()
    assert np.isclose(error, 0, atol=1e-5)


def test_padding():
    np.random.seed(7477)
    channels = [1, 1, 1, 1]
    kernel_sz = [3, 3, 3]
    strides = [1, 1, 1]
    pads = [0, 1, 2]
    cnn = pvml.CNN(channels, kernel_sz, strides, pads)
    X = np.random.randn(1, 11, 11, 1)
    A = cnn.forward(X)
    assert A[0].shape[1] == 11
    assert A[1].shape[1] == 9
    assert A[2].shape[1] == 9
    assert A[3].shape[1] == 11
    Y = np.zeros(X.shape[0], dtype=int)
    D = cnn.backward(Y, A)
    assert A[1].shape[1] == D[0].shape[1]
    assert A[2].shape[1] == D[1].shape[1]
    assert A[3].shape[1] == D[2].shape[1]


def make_backward_data():
    np.random.seed(7477)
    m = 3
    h, w = 7, 9
    n = 8
    k = 2
    channels = [n, 4, 4, k]
    kernel_sz = [3, 3, 3]
    strides = [1, 2, 1]
    pads = [0, 1, 2]

    net = CNNTanh(channels, kernel_sz, strides, pads)
    X = np.random.randn(m, h, w, n)
    Y = np.arange(m) % k
    A = net.forward(X)
    D = net.backward(Y, A)
    grad = net.backward_to_input(D[0], X.shape[1], X.shape[2])
    loss = net.loss(Y, A[-1])
    for index in np.ndindex(*X.shape):
        yield (net, X, Y, grad, loss, index)


@pytest.mark.parametrize("net, X, Y, grad, loss, index", make_backward_data())
def test_backward(net, X, Y, grad, loss, index):
    """Test the gradient of the loss wrt the input."""
    eps = 1e-7
    backup = X[index]
    X[index] += eps
    A1 = net.forward(X)
    loss1 = net.loss(Y, A1[-1])
    ratio = (loss1 - loss) / eps
    assert np.isclose(grad[index], ratio)
    X[index] = backup


def make_parameter_gradients_data():
    np.random.seed(7477)
    m = 2
    h, w = 14, 18
    n = 3
    k = 5
    channels = [n, 4, 4, k]
    kernel_sz = [5, 3, 3]
    strides = [2, 1, 2]
    pads = [2, 1, 0]

    net = CNNTanh(channels, kernel_sz, strides, pads)
    X = np.random.randn(m, h, w, n)
    Y = np.arange(m) % k
    A = net.forward(X)
    D = net.backward(Y, A)
    loss = net.loss(Y, A[-1])
    grad_Ws, grad_bs = net.parameters_gradient(A, D)
    for ps, grad_ps, name in ((net.weights, grad_Ws, "W"), (net.biases, grad_bs, "b")):
        for p, grad_p, l in zip(ps, grad_ps, range(len(ps))):
            for index in np.ndindex(*p.shape):
                yield (net, X, Y, name, p, grad_p, loss, index)


@pytest.mark.parametrize("net, X, Y, name, p, grad_p, loss, index",
                         make_parameter_gradients_data())
def test_parameter_gradients(net, X, Y, name, p, grad_p, loss, index):
    """Test the gradient of the loss wrt the parameters."""
    eps = 1e-7
    backup = p[index]
    p[index] += eps
    A1 = net.forward(X)
    loss1 = net.loss(Y, A1[-1])
    ratio = (loss1 - loss) / eps
    assert np.isclose(grad_p[index], ratio)
    p[index] = backup


def test_train():
    np.random.seed(7477)
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    net = pvml.CNN([X.shape[1], 2], [1])
    X = X.reshape(X.shape[0], 1, 1, X.shape[1])
    net.train(X, Y, lr=1e-1, steps=1000)
    Yhat, P = net.inference(X)
    assert np.all(Y == Yhat)


def test_batch():
    np.random.seed(7477)
    X, Y = test_data.separable_circle_data_set(10, 2)
    net = pvml.CNN([X.shape[1], 2, 2], [1, 1])
    X = X.reshape(X.shape[0], 1, 1, X.shape[1])
    net.train(X, Y, lr=1e-1, steps=1000, batch=10)
    Yhat, P = net.inference(X)
    assert np.all(Y == Yhat)


def test_saveload():
    np.random.seed(7477)
    channels = [2, 3, 4, 5]
    net1 = pvml.CNN(channels)
    f = io.BytesIO()
    net1.save(f)
    f.seek(0)
    net2 = pvml.CNN.load(f)
    for w1, w2 in zip(net1.weights, net2.weights):
        assert np.all(w1 == w2)
    for b1, b2 in zip(net1.biases, net2.biases):
        assert np.all(b1 == b2)
