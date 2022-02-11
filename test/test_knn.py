import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(1, 5))
def test_knn_1(k):
    X, Y = test_data.non_separable_checkerboard_data_set(48, k)
    Yhat, P = pvml.knn_inference(X, X, Y, k=1)
    assert np.all(Y == Yhat)


@pytest.mark.parametrize("k", range(1, 5))
def test_knn_5(k):
    X, Y = test_data.separable_circle_data_set(48, k)
    Yhat, P = pvml.knn_inference(X, X, Y, k=5)
    assert np.all(Y == Yhat)


@pytest.mark.parametrize("k", range(1, 5))
def test_knn_auto_k(k):
    X, Y = test_data.separable_circle_data_set(24, k)
    k, _ = pvml.knn_select_k(X, Y, maxk=7)
    assert k > 0
    assert k < 8
