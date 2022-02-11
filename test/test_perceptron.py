import numpy as np
import pvml
import test_data


def test_train1():
    X, Y = test_data.separable_circle_data_set(50, 2)
    w, b = pvml.perceptron_train(X, Y)
    Yhat, Z = pvml.perceptron_inference(X, w, b)
    assert np.all(Y == Yhat)


def test_train2():
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    w, b = pvml.perceptron_train(X, Y)
    Yhat, Z = pvml.perceptron_inference(X, w, b)
    assert np.all(Y == Yhat)


def test_train3():
    X, Y = test_data.non_separable_checkerboard_data_set(51, 2)
    w, b = pvml.perceptron_train(X, Y, steps=100)
    Yhat, Z = pvml.perceptron_inference(X, w, b)
    assert (Yhat == Y).sum() > 25
