import numpy as np
import pvml
import io
import pytest


def make_BasicgradientW_data():
    np.random.seed(7477)
    n, h, m, t = 3, 2, 4, 5
    init = np.zeros((m, h))
    cell = pvml.RNNBasicCell(n, h)
    X = np.random.randn(m, t, n)
    H = cell.forward(X, np.zeros((m, h)))
    L = H.sum()
    cell.backward(np.ones_like(H))
    gradients = cell.parameters_grad()
    for p in range(len(gradients)):
        for index in np.ndindex(*gradients[p].shape):
            yield (cell, X, init, L, p, gradients, index)


@pytest.mark.parametrize("cell, X, init, L, p, gradients, index",
                         make_BasicgradientW_data())
def test_BasicgradientW(cell, X, init, L, p, gradients, index):
    """Test the gradient of the loss wrt the parameters.

    The loss here is just the sum of all the cell states.
    """
    eps = 1e-7
    backup = cell.parameters()[p][index]
    cell.parameters()[p][index] += eps
    H1 = cell.forward(X, init)
    L1 = H1.sum()
    D = (L1 - L) / eps
    assert np.isclose(gradients[p][index], D)
    cell.parameters()[p][index] = backup


def make_gradient_single_label_data():
    np.random.seed(7477)
    neurons = [3, 2, 4, 2, 3]
    m, t = 3, 5
    rnn = pvml.RNN(neurons)
    X = np.random.randn(m, t, neurons[0])
    Y = np.arange(m) % neurons[-1]
    P = rnn.forward(X)
    L = rnn.loss(Y, P)
    DX = rnn.backward(Y)
    for index in np.ndindex(*X.shape):
        yield (rnn, X, Y, L, DX, index)


@pytest.mark.parametrize("rnn, X, Y, L, DX, index",
                         make_gradient_single_label_data())
def test_gradient_single_label(rnn, X, Y, L, DX, index):
    eps = 1e-7
    backup = X[index]
    X[index] += eps
    P1 = rnn.forward(X)
    L1 = rnn.loss(Y, P1)
    D = (L1 - L) / eps
    assert np.isclose(DX[index], D)
    X[index] = backup


def make_gradient_sequence_data():
    np.random.seed(7477)
    neurons = [3, 2, 4, 2, 3]
    m, t = 3, 5
    rnn = pvml.RNN(neurons)
    X = np.random.randn(m, t, neurons[0])
    Y = np.arange(m * t).reshape(m, t) % neurons[-1]
    P = rnn.forward(X)
    L = rnn.loss(Y, P)
    DX = rnn.backward(Y)
    for index in np.ndindex(*X.shape):
        yield (rnn, X, Y, L, DX, index)


@pytest.mark.parametrize("rnn, X, Y, L, DX, index",
                         make_gradient_sequence_data())
def test_gradient_sequence(rnn, X, Y, L, DX, index):
    eps = 1e-7
    backup = X[index]
    X[index] += eps
    P1 = rnn.forward(X)
    L1 = rnn.loss(Y, P1)
    D = (L1 - L) / eps
    assert np.isclose(DX[index], D)
    X[index] = backup


def test_train():
    np.random.seed(7477)
    k = 4
    neurons = [k, 2 * k, k]
    # Train to echo output a set sequence
    Y = np.arange(k).reshape(1, k)
    X = np.eye(k).reshape(1, k, k)
    rnn = pvml.RNN(neurons)
    rnn.train(X, Y, lr=0.01, steps=100)
    P = rnn.forward(X)
    Z = P[0, :, :].argmax(-1)
    assert np.all(Z == Y[0, :])


def test_saveload():
    np.random.seed(7477)
    neurons = [2, 3, 4, 5]
    rnn = pvml.RNN(neurons)
    f = io.BytesIO()
    rnn.save(f)
    f.seek(0)
    rnn2 = pvml.RNN.load(f)
    assert np.all(rnn.b == rnn2.b)


def test_GRU_forward():
    np.random.seed(7477)
    n, h, m, t = 5, 3, 2, 4
    init = np.zeros((m, h))
    cell = pvml.GRUCell(n, h)
    X = np.random.randn(m, t, n)
    H = cell.forward(X, init)
    assert H.shape == (m, t, h)


def make_gradientX_data():
    np.random.seed(7477)
    n, h, m, t = 2, 2, 2, 3
    init = np.zeros((m, h))
    cell = pvml.GRUCell(n, h)
    X = np.random.randn(m, t, n)
    H = cell.forward(X, init)
    DL = np.ones_like(H)
    DX = cell.backward(DL)
    L0 = H.sum()
    for index in np.ndindex(*X.shape):
        yield (cell, X, DX, init, L0, index)


@pytest.mark.parametrize("cell, X, DX, init, L0, index",
                         make_gradientX_data())
def test_gradientX(cell, X, DX, init, L0, index):
    eps = 1e-7
    backup = X[index]
    X[index] += eps
    H1 = cell.forward(X, init)
    L1 = H1.sum()
    D = (L1 - L0) / eps
    assert np.isclose(DX[index], D)
    X[index] = backup


def make_GRUgradientW_data():
    np.random.seed(7477)
    n, h, m, t = 3, 2, 4, 5
    init = np.zeros((m, h))
    cell = pvml.GRUCell(n, h)
    X = np.random.randn(m, t, n)
    H = cell.forward(X, np.zeros((m, h)))
    L = H.sum()
    cell.backward(np.ones_like(H))
    gradients = cell.parameters_grad()
    for p in range(len(gradients)):
        for index in np.ndindex(*gradients[p].shape):
            yield (cell, X, init, p, gradients, L, index)


@pytest.mark.parametrize("cell, X, init, p, gradients, L, index",
                         make_GRUgradientW_data())
def test_GRUgradientW(cell, X, init, p, gradients, L, index):
    """Test the gradient of the loss wrt the parameters.

    The loss here is just the sum of all the cell states.
    """
    eps = 1e-7
    backup = cell.parameters()[p][index]
    cell.parameters()[p][index] += eps
    H1 = cell.forward(X, init)
    L1 = H1.sum()
    D = (L1 - L) / eps
    assert np.isclose(gradients[p][index], D)
    cell.parameters()[p][index] = backup


def test_abstract_forward():
    n, h, m, t = 3, 2, 4, 5
    init = np.zeros((m, h))
    cell = pvml.RNNAbstractCell()
    X = np.zeros((m, t, n))
    with pytest.raises(NotImplementedError):
        cell.forward(X, init)


def test_backward():
    n, m, t = 3, 4, 5
    D = np.zeros((m, t, n))
    cell = pvml.RNNAbstractCell()
    with pytest.raises(NotImplementedError):
        cell.backward(D)


def test_parameters():
    cell = pvml.RNNAbstractCell()
    with pytest.raises(NotImplementedError):
        cell.parameters()


def test_parameters_grad():
    cell = pvml.RNNAbstractCell()
    with pytest.raises(NotImplementedError):
        cell.parameters_grad()
