import numpy as np
import pvml
import test_data
import pytest


def test_train1():
    ada = pvml.AdaBoost()
    X, Y = test_data.separable_circle_data_set(50, 2)
    ada.train(X, Y, 10)
    Yhat = ada.inference(X)[0]
    assert np.all(Y == Yhat)


def test_train2():
    ada = pvml.AdaBoost()
    X, Y = test_data.separable_stripes_data_set(50, 2)
    ada.train(X, Y, 10)
    Yhat = ada.inference(X)[0]
    assert np.all(Y == Yhat)


def test_no_split():
    ada = pvml.AdaBoost()
    X = np.ones((100, 2))
    Y = np.arange(100) % 2
    ada.train(X, Y, 10)
    Yhat = ada.inference(X)[0]
    assert(ada.size() == 0)
    assert(Yhat.shape == Y.shape)


def test_minsize():
    ada = pvml.AdaBoost()
    X, Y = test_data.separable_stripes_data_set(50, 2)
    ada.train(X, Y, 10, minsize=30)
    assert(ada.size() == 0)


def test_wrong_dimensions():
    ada = pvml.AdaBoost()
    X, Y = test_data.non_separable_checkerboard_data_set(50, 2)
    ada.train(X, Y, 10)
    X1 = np.ones((50, 1))
    print(ada.indices)
    with pytest.raises(ValueError):
        ada.inference(X1)
