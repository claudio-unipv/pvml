import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(1, 6))
def test_categorical(k):
    X, Y = test_data.categorical_data_set(50, k)
    probs, priors = pvml.categorical_naive_bayes_train(X, Y)
    Yhat, scores = pvml.categorical_naive_bayes_inference(X, probs, priors)
    assert np.all(Y == Yhat)


def test_categorical2():
    X, Y = test_data.categorical_data_set(50, 2)
    priors = np.array([0.0, 1.0])
    probs, priors = pvml.categorical_naive_bayes_train(X, Y, priors)
    assert np.all(priors == np.array([0.0, 1.0]))
    Yhat, _ = pvml.categorical_naive_bayes_inference(X, probs, priors)
    assert np.all(Yhat == np.ones_like(Yhat))


@pytest.mark.parametrize("k", range(1, 6))
def test_multinomial(k):
    X, Y = test_data.bow_data_set(50, k)
    w, b = pvml.multinomial_naive_bayes_train(X, Y)
    Yhat, scores = pvml.multinomial_naive_bayes_inference(X, w, b)
    assert np.all(Y == Yhat)


def test_multinomial2():
    X, Y = test_data.bow_data_set(50, 2)
    priors = np.array([0.0, 1.0])
    w, b = pvml.multinomial_naive_bayes_train(X, Y, priors)
    Yhat, _ = pvml.multinomial_naive_bayes_inference(X, w, b)
    assert np.all(Yhat == np.ones_like(Yhat))


@pytest.mark.parametrize("k", range(1, 6))
def test_gaussian(k):
    X, Y = test_data.separable_disks_data_set(50, k)
    ms, vs, ps = pvml.gaussian_naive_bayes_train(X, Y)
    Yhat, scores = pvml.gaussian_naive_bayes_inference(X, ms, vs, ps)
    assert np.all(Y == Yhat)


def test_gaussian2():
    X, Y = test_data.separable_hypercubes_data_set(50, 2)
    priors = np.array([0.0, 1.0])
    ms, vs, priors = pvml.gaussian_naive_bayes_train(X, Y, priors)
    Yhat, _ = pvml.gaussian_naive_bayes_inference(X, ms, vs, priors)
    assert np.all(Yhat == np.ones_like(Yhat))
