import numpy as np
import unittest


def separable_circle_data_set(n, k):
    ii = np.arange(n)
    a = (np.pi / 5) + 2 * np.pi * ii / n
    Y = (k * ii) // n
    X = np.stack([np.cos(a) + 0.5, np.sin(a) - 0.5], 1)
    return X, Y


def separable_hypercubes_data_set(n, k):
    X = np.random.random((n, k))
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
    Y = np.arange(n) % k
    Z = np.arange(k * 3) // 3
    P = 0.1 + 0.8 * (Z[None, :, None] == Y[:, None, None])
    X = (np.random.random((n, k * 3, 5)) < P).sum(2)
    return X, Y


class TestTestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_separable_circle(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = separable_circle_data_set(4 * k, k)
                c = np.bincount(Y, minlength=k)
                self.assertEqual(c.min(), c.max())

    def test_separable_hypercubes(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = separable_hypercubes_data_set(4 * k, k)
                c = np.bincount(Y, minlength=k)
                self.assertEqual(c.min(), c.max())

    def test_separable_stripes(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = separable_stripes_data_set(4 * k, k)
                self.assertEqual(Y.min(), 0)
                self.assertEqual(Y.max(), k - 1)

    def test_non_separable_checkerboard(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = non_separable_checkerboard_data_set(4 * k, k)
                self.assertEqual(Y.min(), 0)
                self.assertEqual(Y.max(), k - 1)

    def test_categorical(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = categorical_data_set(4 * k, k)
                self.assertEqual(Y.min(), 0)
                self.assertEqual(Y.max(), k - 1)
                self.assertEqual(np.abs(np.modf(X)[0]).max(), 0)

    def test_bow(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = bow_data_set(4 * k, k)
                self.assertEqual(Y.min(), 0)
                self.assertEqual(Y.max(), k - 1)
                self.assertEqual(np.abs(np.modf(X)[0]).max(), 0)


if __name__ == '__main__':
    unittest.main()
