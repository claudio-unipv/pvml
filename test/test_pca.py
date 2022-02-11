import pvml
import numpy as np


def test_pca1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.pca(X)
    assert np.allclose(X1.mean(0), np.zeros(2))
    assert np.allclose(X1.var(0), X.var(0))


def test_pca2():
    a = np.arange(10)
    X = np.tile(a, (7, 1))
    T = np.ones((10, 10))
    X1, T1 = pvml.pca(X, T)
    assert X1.shape[1] == 1
    assert np.isclose(X1.mean(0), 0)
    assert np.all(T1[:, 0] == np.ones(10))
