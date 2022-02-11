import numpy as np
import pvml
import test_data
import pytest


@pytest.mark.parametrize("k", range(1, 6))
def test_train_no_pruning(k):
    tree = pvml.ClassificationTree()
    div = ["gini", "entropy", "error"]
    X, Y = test_data.separable_circle_data_set(50, k)
    tree.train(X, Y, diversity=div[k % 3], pruning_cv=0)
    Yhat = tree.inference(X)[0]
    assert np.all(Y == Yhat)


@pytest.mark.parametrize("k", range(1, 6))
def test_train_pruning(k):
    tree = pvml.ClassificationTree()
    div = ["gini", "entropy", "error"]
    X, Y = test_data.separable_hypercubes_data_set(51, k)
    tree.train(X, Y, diversity=div[k % 3], pruning_cv=5)
    Yhat = tree.inference(X)[0]
    assert np.all(Y == Yhat)


def test_dump():
    tree = pvml.ClassificationTree()
    X, Y = test_data.separable_hypercubes_data_set(21, 3)
    tree.train(X, Y)
    s = tree._dumps()
    assert len(s.splitlines()) > 3


def test_check1():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    Y = np.arange(5)
    tree = pvml.ClassificationTree()
    tree.train(X, Y, pruning_cv=0)
    with pytest.raises(ValueError):
        tree.inference(np.arange(5))


def test_check2():
    X = np.linspace(-1, 1, 10).reshape(5, 2)
    X[:, 0] = 0
    Y = np.arange(5)
    tree = pvml.ClassificationTree()
    tree.train(X, Y, pruning_cv=0)
    with pytest.raises(ValueError):
        X = np.linspace(-1, 1, 5).reshape(5, 1)
        tree.inference(X)
