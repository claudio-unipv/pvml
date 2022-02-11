import numpy as np
import pvml
import test_data


def test_inference():
    w = [1, -1]
    X = [[0, 0], [-1, 1]]
    b = 1
    Yhat, Z = pvml.svm_inference(X, w, b)
    assert Yhat[0] == 1
    assert Yhat[1] == 0
    assert Z[0] == 1
    assert Z[1] == -1


def test_train1():
    X, Y = test_data.separable_circle_data_set(50, 2)
    w, b = pvml.svm_train(X, Y, 0, lr=1e-1, steps=1000)
    Yhat, P = pvml.svm_inference(X, w, b)
    assert np.all(Y == Yhat)
    assert np.all(Yhat == (P > 0))


def test_train2():
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    w, b = pvml.svm_train(X, Y, 0.0001, lr=10, steps=1000)
    Yhat, P = pvml.svm_inference(X, w, b)
    assert np.all(Y == Yhat)
    assert np.all(Yhat == (P > 0))


def test_train3():
    X, Y = test_data.separable_stripes_data_set(50, 2)
    w, b = pvml.svm_train(X, Y, 0.0001, lr=10, steps=1000)
    Yhat, P = pvml.svm_inference(X, w, b)
    assert np.all(Y == Yhat)
    assert np.all(Yhat == (P > 0))


def test_hinge_loss1():
    Y = [0, 1]
    Z = [-1, 1]
    loss = pvml.hinge_loss(Y, Z)
    assert loss == 0


def test_hinge_loss2():
    Y = [0, 1]
    Z = [0, 0]
    loss = pvml.hinge_loss(Y, Z)
    assert loss == 1


def test_hinge_loss3():
    Y = [0, 1]
    Z = [1, -1]
    loss = pvml.hinge_loss(Y, Z)
    assert loss == 2
