import numpy as np
import pvml
import test_data


def test_inference():
    w = [1, -1]
    X = [[-1, 1]]
    b = 2
    p = pvml.logreg_inference(X, w, b)
    assert np.all(p == 0.5)


def test_train_l2():
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    w, b = pvml.logreg_train(X, Y, 0.0001, lr=10, steps=1000)
    P = pvml.logreg_inference(X, w, b)
    Yhat = (P > 0.5).astype(int)
    assert np.all(Y == Yhat)


def test_train_l1():
    X, Y = test_data.separable_stripes_data_set(50, 2)
    w, b = pvml.logreg_l1_train(X, Y, 0.0001, lr=10, steps=1000)
    P = pvml.logreg_inference(X, w, b)
    Yhat = (P > 0.5).astype(int)
    assert np.all(Y == Yhat)


def test_cross_entropy1():
    Y = [0, 0, 1, 1]
    e = pvml.binary_cross_entropy(Y, Y)
    assert e == 0


def test_cross_entropy2():
    Y1 = [0, 0, 1, 1]
    Y2 = [1, 1, 0, 0]
    e = pvml.binary_cross_entropy(Y1, Y2)
    assert np.isinf(e)
