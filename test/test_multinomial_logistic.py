import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(1, 6))
def test_train1(k):
    X, Y = test_data.separable_circle_data_set(50, k)
    w, b = pvml.multinomial_logreg_train(X, Y, 0, lr=1e-1, steps=1000)
    P = pvml.multinomial_logreg_inference(X, w, b)
    Yhat = P.argmax(1)
    assert np.all(Y == Yhat)


@pytest.mark.parametrize("k", range(1, 6))
def test_train2(k):
    X, Y = test_data.separable_hypercubes_data_set(49, k)
    w, b = pvml.multinomial_logreg_train(X, Y, 1e-4, lr=1, steps=10000)
    P = pvml.multinomial_logreg_inference(X, w, b)
    Yhat = P.argmax(1)
    assert np.all(Y == Yhat)


def test_softmax():
    Z = np.random.randn(20, 4)
    P = pvml.softmax(Z)
    error = np.abs(P.sum(1) - 1).max()
    assert np.isclose(error, 0)


def test_one_hot_vectors():
    k = 7
    Y = np.arange(20) % k
    H = pvml.one_hot_vectors(Y, k)
    assert np.all(H.sum(1) == np.ones(20))
    assert np.all(np.bincount(Y, minlength=k) == H.sum(0))


def test_cross_entropy():
    P = [[1, 0, 0]]
    Y = [0]
    ce = pvml.cross_entropy(Y, P)
    assert np.isclose(ce, 0)
