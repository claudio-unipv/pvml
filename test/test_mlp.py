import numpy as np
import pvml
import io
import test_data
import pytest


class MLPTanh(pvml.MLP):
    """MLP with hyperbolic tangent activation function.

    It is used here because tanh is more reliable than ReLU for
    numerical testing.
    """
    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return np.tanh(X)

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        return d * (1 - Y ** 2)


def make_parameter_gradients_data():
    np.random.seed(7477)
    m = 4
    n = 3
    k = 5
    net = MLPTanh([n, 2, 7, k])
    X = np.random.randn(m, n)
    Y = np.arange(m) % k
    A = net.forward(X)
    D = net.backward(Y, A)
    loss = net.loss(Y, A[-1])
    grad_Ws, grad_bs = net.parameters_gradient(A, D)
    for ps, grad_ps, name in ((net.weights, grad_Ws, "W"), (net.biases, grad_bs, "b")):
        for p, grad_p, l in zip(ps, grad_ps, range(len(ps))):
            for index in np.ndindex(*p.shape):
                yield (net, X, Y, loss, name, p, grad_p, index)


@pytest.mark.parametrize("net, X, Y, loss, name, p, grad_p, index",
                         make_parameter_gradients_data())
def test_parameter_gradients(net, X, Y, loss, name, p, grad_p, index):
    """Test the gradient of the loss wrt the parameters."""
    eps = 1e-7
    backup = p[index]
    p[index] += eps
    A1 = net.forward(X)
    loss1 = net.loss(Y, A1[-1])
    ratio = (loss1 - loss) / eps
    assert np.isclose(grad_p[index], ratio)
    p[index] = backup


def make_backward_data():
    np.random.seed(7477)
    m = 3
    n = 2
    k = 3
    net = MLPTanh([n, 2, 3, 3, 2, k])
    X = np.random.randn(m, n)
    Y = np.arange(m) % k
    A = net.forward(X)
    loss = net.loss(Y, A[-1])
    D = net.backward(Y, A)
    grad = net.backward_to_input(D[0])
    for index in np.ndindex(*X.shape):
        yield (net, X, Y, grad, loss, index)


@pytest.mark.parametrize("net, X, Y, grad, loss, index", make_backward_data())
def test_backward(net, X, Y, grad, loss, index):
    eps = 1e-7
    backup = X[index]
    X[index] += eps
    A1 = net.forward(X)
    loss1 = net.loss(Y, A1[-1])
    ratio = (loss1 - loss) / eps
    assert np.isclose(grad[index], ratio)
    X[index] = backup


def test_train():
    np.random.seed(7477)
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    net = pvml.MLP([X.shape[1], 2])
    net.train(X, Y, lr=1e-1, steps=1000)
    Yhat, P = net.inference(X)
    assert np.all(Y == Yhat)


def test_batch():
    np.random.seed(7477)
    X, Y = test_data.separable_circle_data_set(50, 2)
    net = pvml.MLP([X.shape[1], 2, 2])
    net.train(X, Y, lr=1e-1, steps=1000, batch=10)
    Yhat, P = net.inference(X)
    assert np.all(Y == Yhat)


def test_saveload():
    np.random.seed(7477)
    neurons = [2, 3, 4, 5]
    net1 = pvml.MLP(neurons)
    f = io.BytesIO()
    net1.save(f)
    f.seek(0)
    net2 = pvml.MLP.load(f)
    for w1, w2 in zip(net1.weights, net2.weights):
        assert np.all(w1 == w2)
    for b1, b2 in zip(net1.biases, net2.biases):
        assert np.all(b1 == b2)
