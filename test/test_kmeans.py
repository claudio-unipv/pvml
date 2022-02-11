import pvml
import numpy as np
import pytest


def normalize_labels(Y):
    # Change labels by permuting them in sych a way that their first
    # occurrences are sorted.  This makes it possible to compare
    # algorithms that are not consistent in assigning labels (like in
    # clustering).
    #
    # (3, 1, 1, 0, 2, 0) -> (0, 1, 1, 2, 3, 2)
    _, index = np.unique(Y, return_index=True)
    old_labels = np.argsort(index)  # e.g. [3, 1, 0, 2]
    new_labels = np.argsort(old_labels)  # e.g. [2, 1, 3, 0]
    return new_labels[Y]


@pytest.mark.parametrize("k", range(1, 12))
def test_one_per_class(k):
    X = np.random.randn(k, 3)
    centroids = pvml.kmeans_train(X, k)
    Y, _ = pvml.kmeans_inference(X, centroids)
    assert np.all(normalize_labels(Y) == np.arange(k))


@pytest.mark.parametrize("k", range(1, 5))
def test_linear_clusters(k):
    Y = np.arange(k)
    a = np.linspace(0, 2 * np.pi, k)
    X = np.stack([np.cos(a) + 10 * Y, np.sin(a)], 1)
    centroids = pvml.kmeans_train(X, k)
    Z, _ = pvml.kmeans_inference(X, centroids)
    assert np.all(normalize_labels(Y) == normalize_labels(Z))


def test_respawn():
    m = 100
    k = 5
    X = np.random.randn(m, 2)
    centroids = np.zeros((k, 2))
    centroids = pvml.kmeans_train(X, k, init_centroids=centroids)
    Z, _ = pvml.kmeans_inference(X, centroids)
    assert np.all(np.unique(Z) == np.arange(k))


def test_init():
    m = 100
    k = 5
    X = np.random.randn(m, 2)
    init_centroids = np.arange(k * 2).reshape(k, 2)
    centroids = pvml.kmeans_train(X, k, init_centroids=init_centroids, steps=0)
    assert np.all(init_centroids == centroids)


def test_errors1():
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        pvml.kmeans_train(X, 0)


def test_errors2():
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        pvml.kmeans_train(X, 3, init_centroids=[[0, 0], [0, 0]])


def test_errors3():
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        pvml.kmeans_train(X, 3, init_centroids=[[1], [1], [1]])


def test_errors4():
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        pvml.kmeans_train(X, 11)
