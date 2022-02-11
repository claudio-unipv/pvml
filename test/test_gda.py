import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(1, 5))
def test_heteroscedastic_gda(k):
    X, Y = test_data.separable_circle_data_set(48, k)
    means, icovs, priors = pvml.hgda_train(X, Y)
    Yhat, scores = pvml.hgda_inference(X, means, icovs, priors)
    assert np.all(Y == Yhat)
    assert np.all(priors == np.ones(k) / k)


def test_heteroscedastic_priors():
    k = 3
    X, Y = test_data.non_separable_checkerboard_data_set(24, k)
    means, icovs, priors = pvml.hgda_train(X, Y, priors=np.array([1, 0, 0]))
    Yhat, scores = pvml.hgda_inference(X, means, icovs, priors)
    assert np.all(Yhat == np.zeros(24))


@pytest.mark.parametrize("k", range(1, 5))
def test_omoscedastic_gda(k):
    X, Y = test_data.separable_stripes_data_set(48, k)
    w, b = pvml.ogda_train(X, Y)
    Yhat, scores = pvml.ogda_inference(X, w, b)
    assert np.all(Y == Yhat)


def test_omoscedastic_priors():
    k = 4
    X, Y = test_data.non_separable_checkerboard_data_set(24, k)
    w, b = pvml.ogda_train(X, Y, priors=np.array([1, 0, 0, 0]))
    Yhat, scores = pvml.ogda_inference(X, w, b)
    assert np.all(Yhat == np.zeros(24))


@pytest.mark.parametrize("k", range(1, 5))
def test_mindist(k):
    X, Y = test_data.separable_circle_data_set(48, k)
    means = pvml.mindist_train(X, Y)
    Yhat, scores = pvml.mindist_inference(X, means)
    assert np.all(Y == Yhat)
