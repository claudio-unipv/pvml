import unittest
import pvml
import numpy as np


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


class TestKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_one_per_class(self):
        for k in range(1, 12):
            with self.subTest(k):
                X = np.random.randn(k, 3)
                centroids = pvml.kmeans_train(X, k)
                Y, _ = pvml.kmeans_inference(X, centroids)
                self.assertListEqual(normalize_labels(Y).tolist(),
                                     list(range(k)))

    def _linear(self, k, m):
        Y = np.arange(m) % k
        a = np.linspace(0, 2 * np.pi, m)
        X = np.stack([np.cos(a) + 10 * Y, np.sin(a)], 1)
        centroids = pvml.kmeans_train(X, k)
        Z, _ = pvml.kmeans_inference(X, centroids)
        self.assertListEqual(normalize_labels(Y).tolist(),
                             normalize_labels(Z).tolist())

    def test_linear_clusters(self):
        for k in range(1, 5):
            with self.subTest(k):
                self._linear(k, k)

    def test_respawn(self):
        m = 100
        k = 5
        X = np.random.randn(m, 2)
        centroids = np.zeros((k, 2))
        centroids = pvml.kmeans_train(X, k, init_centroids=centroids)
        Z, _ = pvml.kmeans_inference(X, centroids)
        self.assertListEqual(np.unique(Z).tolist(), list(range(k)))

    def test_init(self):
        m = 100
        k = 5
        X = np.random.randn(m, 2)
        init_centroids = np.arange(k * 2).reshape(k, 2)
        centroids = pvml.kmeans_train(X, k, init_centroids=init_centroids, steps=0)
        self.assertListEqual(init_centroids.tolist(), centroids.tolist())

    def test_errors1(self):
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            pvml.kmeans_train(X, 0)

    def test_errors2(self):
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            pvml.kmeans_train(X, 3, init_centroids=np.zeros((2, 2)))

    def test_errors3(self):
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            pvml.kmeans_train(X, 3, init_centroids=np.zeros((3, 1)))

    def test_errors4(self):
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            pvml.kmeans_train(X, 11)


if __name__ == '__main__':
    unittest.main()
