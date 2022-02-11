import numpy as np
import pytest


_RANDOM_SEED = 7477


def separable_circle_data_set(n, k):
    ii = np.arange(n)
    a = (np.pi / 5) + 2 * np.pi * ii / n
    Y = (k * ii) // n
    X = np.stack([np.cos(a) + 0.5, np.sin(a) - 0.5], 1)
    return X, Y


def separable_disks_data_set(n, k):
    rng = np.random.default_rng(_RANDOM_SEED)
    a = np.pi * rng.uniform(size=(n))
    r = rng.uniform(size=(n))
    Y = np.arange(n) % k
    X = np.stack([2 * Y + r * np.cos(a) + 0.5, r * np.sin(a)], 1)
    return X, Y


def separable_hypercubes_data_set(n, k):
    rng = np.random.default_rng(_RANDOM_SEED)
    X = rng.uniform(size=(n, k))
    Y = np.arange(n) % k
    X[np.arange(n), Y] += Y
    return X, Y


def separable_stripes_data_set(n, k):
    rs = np.sqrt(n).astype(int)
    r = np.arange(n) // rs
    c = np.arange(n) % rs
    v1 = np.array([1, 0.1])
    v2 = np.array([-0.1, 1])
    X = np.outer(r, v1) + np.outer(c, v2)
    Y = (k * r) // (r.max() + 1)
    return X, Y


def non_separable_checkerboard_data_set(n, k):
    rs = np.sqrt(n).astype(int)
    r = np.arange(n) // rs
    c = np.arange(n) % rs
    v1 = np.array([0.7, -0.2])
    v2 = np.array([0.2, -0.7])
    X = np.outer(r, v1) + np.outer(c, v2)
    Y = (r + c) % k
    return X, Y


def categorical_data_set(n, k):
    Y = np.arange(n) % k
    X = Y[:, None] % np.array([2, 3, k])
    return X, Y


def bow_data_set(n, k):
    rng = np.random.default_rng(_RANDOM_SEED)
    Y = np.arange(n) % k
    Z = np.arange(k * 3) // 3
    P = 0.1 + 0.8 * (Z[None, :, None] == Y[:, None, None])
    X = (rng.uniform(size=(n, k * 3, 5)) < P).sum(2)
    return X, Y


@pytest.mark.parametrize("k", range(1, 6))
def test_separable_circle(k):
    X, Y = separable_circle_data_set(4 * k, k)
    c = np.bincount(Y, minlength=k)
    assert c.min() == c.max()


@pytest.mark.parametrize("k", range(1, 6))
def test_separable_disks(k):
    X, Y = separable_disks_data_set(4 * k, k)
    c = np.bincount(Y, minlength=k)
    assert c.min() == c.max()


@pytest.mark.parametrize("k", range(1, 6))
def test_separable_hypercubes(k):
    X, Y = separable_hypercubes_data_set(4 * k, k)
    c = np.bincount(Y, minlength=k)
    assert c.min() == c.max()


@pytest.mark.parametrize("k", range(1, 6))
def test_separable_stripes(k):
    X, Y = separable_stripes_data_set(4 * k, k)
    assert Y.min() == 0
    assert Y.max() == k - 1


@pytest.mark.parametrize("k", range(1, 6))
def test_non_separable_checkerboard(k):
    X, Y = non_separable_checkerboard_data_set(4 * k, k)
    assert Y.min() == 0
    assert Y.max() == k - 1


@pytest.mark.parametrize("k", range(1, 6))
def test_categorical(k):
    X, Y = categorical_data_set(4 * k, k)
    assert Y.min() == 0
    assert Y.max() == k - 1
    assert np.abs(np.modf(X)[0]).max() == 0


@pytest.mark.parametrize("k", range(1, 6))
def test_bow(k):
    X, Y = bow_data_set(4 * k, k)
    assert Y.min() == 0
    assert Y.max() == k - 1
    assert np.abs(np.modf(X)[0]).max() == 0
