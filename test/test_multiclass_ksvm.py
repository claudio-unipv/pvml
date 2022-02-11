import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(2, 6))
def test_train_ovo(k):
    X, Y = test_data.separable_circle_data_set(50, k)
    a, b = pvml.one_vs_one_ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
    Yhat = pvml.one_vs_one_ksvm_inference(X, X, a, b, "rbf", 0.1)[0]
    assert np.all(Y == Yhat)


@pytest.mark.parametrize("k", range(2, 6))
def test_train_ovr(k):
    X, Y = test_data.separable_circle_data_set(50, k)
    a, b = pvml.one_vs_rest_ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
    Yhat = pvml.one_vs_rest_ksvm_inference(X, X, a, b, "rbf", 0.1)[0]
    assert np.all(Y == Yhat)
