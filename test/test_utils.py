import numpy as np
import pvml


def test_log_nowarn1():
    y = pvml.log_nowarn(1)
    assert y == 0


def test_log_nowarn2():
    y = pvml.log_nowarn(0)
    assert y < 0 and np.isinf(y)


def test_one_hot_vectors():
    k = 4
    Y = np.arange(20) % k
    H = pvml.one_hot_vectors(Y, k)
    assert np.all(H.sum(0) == Y.size // k)
    assert np.all(H.sum(1) == 1)


def test_squared_distance_matrix_sym():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    D = pvml.squared_distance_matrix(X, X)
    assert np.all(D >= 0)
    assert np.all(np.diag(D) == 0)


def test_squared_distance_matrix_nosym():
    X1 = np.linspace(-1, 1, 10).reshape(5, 2)
    X2 = np.linspace(-1, 1, 6).reshape(3, 2)
    D = pvml.squared_distance_matrix(X1, X2)
    assert np.all(D >= 0)
    assert D.shape[0] == X1.shape[0] and D.shape[1] == X2.shape[0]
