import numpy as np
import pvml
import test_data
import pytest


def test_train1():
    X, Y = test_data.separable_circle_data_set(50, 2)
    alpha, b = pvml.ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
    Yhat, P = pvml.ksvm_inference(X, X, alpha, b, "rbf", 0.1)
    assert np.all(Y == Yhat)
    assert np.all(Yhat == (P > 0))


def test_train2():
    X, Y = test_data.separable_hypercubes_data_set(12, 2)
    alpha, b = pvml.ksvm_train(X, Y, "polynomial", 2, 0, lr=10, steps=1000)
    Yhat, P = pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)
    assert np.all(Y == Yhat)
    assert np.all(Yhat == (P > 0))


def test_rbf_kernel():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    K = pvml.kernel(X, X, "rbf", 1)
    assert np.all(np.diag(K) == np.ones(5))
    evals = np.linalg.eigvalsh(K)
    assert np.all(evals >= 0)


def test_polynomial_kernel():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    K = pvml.kernel(X, X, "polynomial", 2)
    evals = np.linalg.eigvalsh(K)
    # Here some eigvals may result slightly negative.
    assert np.allclose(np.minimum(evals, 0), 0)


def test_unknown_kernel():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    with pytest.raises(ValueError):
        pvml.kernel(X, X, "unknown", 2)


def test_wrong_dimensions1():
    X1 = np.linspace(-1, 1, 10).reshape(5, 2)
    X2 = np.linspace(-1, 1, 12).reshape(4, 3)
    with pytest.raises(ValueError):
        pvml.kernel(X1, X2, "polynomial", 2)


def test_wrong_dimensions2():
    X1 = np.linspace(-1, 1, 10)
    X2 = np.linspace(-1, 1, 10)
    with pytest.raises(ValueError):
        pvml.kernel(X1, X2, "polynomial", 2)


def test_wrong_coefficients1():
    X = np.linspace(-1, 1, 10)
    alpha = np.linspace(-1, 1, 10)
    b = 0
    with pytest.raises(ValueError):
        pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)


def test_wrong_coefficients2():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    alpha = np.linspace(-1, 1, 10)
    b = 0
    with pytest.raises(ValueError):
        pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)


def test_wrong_coefficients3():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    alpha = np.linspace(-1, 1, 10).reshape(5, 2)
    b = 0
    with pytest.raises(ValueError):
        pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)


def test_wrong_bias():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    alpha = np.linspace(-1, 1, 5)
    b = np.linspace(-1, 1, 5)
    with pytest.raises(ValueError):
        pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)
