import pvml
import numpy as np
import pytest


def test_meanvar1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.meanvar_normalization(X)
    assert np.allclose(X1.mean(0), np.zeros(2))
    assert np.allclose(X1.var(0), np.ones(2))


def test_meanvar2():
    X = np.arange(100).reshape(10, 10)
    Z = X.copy()
    X1, Z1 = pvml.meanvar_normalization(X, Z)
    assert np.isclose(np.abs(X1.mean(0)).max(), 0)
    assert np.isclose(np.abs(X1.var(0) - 1).max(), 0)
    assert np.all(X1 == Z1)


def test_minmax1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.minmax_normalization(X)
    assert np.allclose(X1.min(0), np.zeros(2))
    assert np.allclose(X1.max(0), np.ones(2))


def test_minmax2():
    X = np.arange(100).reshape(10, 10)
    Z = X.copy()
    X1, Z1 = pvml.minmax_normalization(X, Z)
    assert np.allclose(X1.min(0), np.zeros(10))
    assert np.allclose(X1.max(0), np.ones(10))
    assert np.all(X1 == Z1)


def test_maxabs1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.maxabs_normalization(X)
    assert np.allclose(np.abs(X1).max(0), np.ones(2))


def test_maxabs2():
    X = np.arange(100).reshape(10, 10)
    Z = X.copy()
    X1, Z1 = pvml.maxabs_normalization(X, Z)
    assert np.allclose(np.abs(X1).max(0), np.ones(10))
    assert np.all(X1 == Z1)


def test_l2_1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.l2_normalization(X)
    assert np.isclose(np.abs((X1 ** 2).sum(1) - 1).max(), 0)


def test_l2_2():
    X = np.arange(100).reshape(10, 10)
    Z = X.copy()
    X1, Z1 = pvml.l2_normalization(X, Z)
    assert np.isclose(np.abs((X1 ** 2).sum(1) - 1).max(), 0)
    assert np.all(X1 == Z1)


def test_l1_1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.l1_normalization(X)
    assert np.isclose(np.abs(np.abs(X1).sum(1) - 1).max(), 0)


def test_l1_2():
    X = np.arange(100).reshape(10, 10)
    Z = X.copy()
    X1, Z1 = pvml.l1_normalization(X, Z)
    assert np.isclose(np.abs(np.abs(X1).sum(1) - 1).max(), 0)
    assert np.all(X1 == Z1)


def test_whitening1():
    a = 2 * np.pi * np.arange(50) / 50
    X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
    X1 = pvml.whitening(X)
    assert np.allclose(X1.mean(0), np.zeros(2))
    c = np.cov(X1.T)
    assert np.allclose(c, np.eye(2))


def test_whitening2():
    X = (np.arange(30) % 4).reshape(10, 3)
    T = np.ones((5, 3))
    X1, T1 = pvml.whitening(X, T)
    assert np.isclose(np.abs(X1.mean(0)).max(), 0)
    c = np.cov(X1.T)
    assert np.isclose(np.abs(c - np.eye(3)).max(), 0)
    assert np.all(T1.std(0) == np.zeros(3))


def test_wrong_dimensions1():
    X = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        pvml.meanvar_normalization(X)


def test_wrong_dimensions2():
    X1 = np.linspace(0, 1, 10).reshape(5, 2)
    X2 = np.linspace(0, 1, 12).reshape(4, 3)
    with pytest.raises(ValueError):
        pvml.meanvar_normalization(X1, X2)
