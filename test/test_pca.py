import unittest
import pvml
import numpy as np


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_pca1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.pca(X)
        self.assertAlmostEqual(X1.mean(0)[0], 0)
        self.assertAlmostEqual(X1.mean(0)[1], 0)
        self.assertAlmostEqual(X1.var(0)[0], X.var(0)[0])
        self.assertAlmostEqual(X1.var(0)[1], X.var(0)[1])

    def test_pca2(self):
        a = np.arange(10)
        X = np.tile(a, (7, 1))
        T = np.ones((10, 10))
        X1, T1 = pvml.pca(X, T)
        self.assertEqual(X1.shape[1], 1)
        self.assertAlmostEqual(X1.mean(0), 0)
        self.assertEqual(T1[:, 0].tolist(), [1] * 10)


if __name__ == '__main__':
    unittest.main()
