import unittest
import pvml
import numpy as np


class TestNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_meanvar1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.meanvar_normalization(X)
        self.assertAlmostEqual(X1.mean(0)[0], 0)
        self.assertAlmostEqual(X1.mean(0)[1], 0)
        self.assertAlmostEqual(X1.var(0)[0], 1)
        self.assertAlmostEqual(X1.var(0)[1], 1)

    def test_meanvar2(self):
        X = np.arange(100).reshape(10, 10)
        Z = X.copy()
        X1, Z1 = pvml.meanvar_normalization(X, Z)
        self.assertAlmostEqual(np.abs(X1.mean(0)).max(), 0)
        self.assertAlmostEqual(np.abs(X1.var(0) - 1).max(), 0)
        self.assertListEqual(X1.tolist(), Z1.tolist())

    def test_minmax1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.minmax_normalization(X)
        self.assertAlmostEqual(X1.min(0)[0], 0)
        self.assertAlmostEqual(X1.max(0)[1], 1)
        self.assertAlmostEqual(X1.min(0)[0], 0)
        self.assertAlmostEqual(X1.max(0)[1], 1)

    def test_minmax2(self):
        X = np.arange(100).reshape(10, 10)
        Z = X.copy()
        X1, Z1 = pvml.minmax_normalization(X, Z)
        self.assertAlmostEqual(X1.min(0)[0], 0)
        self.assertAlmostEqual(X1.max(0)[1], 1)
        self.assertAlmostEqual(X1.min(0)[0], 0)
        self.assertAlmostEqual(X1.max(0)[1], 1)
        self.assertListEqual(X1.tolist(), Z1.tolist())

    def test_maxabs1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.maxabs_normalization(X)
        self.assertAlmostEqual(np.abs(X1).max(0)[0], 1)
        self.assertAlmostEqual(np.abs(X1).max(0)[1], 1)

    def test_maxabs2(self):
        X = np.arange(100).reshape(10, 10)
        Z = X.copy()
        X1, Z1 = pvml.maxabs_normalization(X, Z)
        self.assertAlmostEqual(np.abs(X1).max(0)[0], 1)
        self.assertAlmostEqual(np.abs(X1).max(0)[1], 1)
        self.assertListEqual(X1.tolist(), Z1.tolist())

    def test_l2_1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.l2_normalization(X)
        self.assertAlmostEqual(np.abs((X1 ** 2).sum(1) - 1).max(), 0)

    def test_l2_2(self):
        X = np.arange(100).reshape(10, 10)
        Z = X.copy()
        X1, Z1 = pvml.l2_normalization(X, Z)
        self.assertAlmostEqual(np.abs((X1 ** 2).sum(1) - 1).max(), 0)
        self.assertListEqual(X1.tolist(), Z1.tolist())

    def test_l1_1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.l1_normalization(X)
        self.assertAlmostEqual(np.abs(np.abs(X1).sum(1) - 1).max(), 0)

    def test_l1_2(self):
        X = np.arange(100).reshape(10, 10)
        Z = X.copy()
        X1, Z1 = pvml.l1_normalization(X, Z)
        self.assertAlmostEqual(np.abs(np.abs(X1).sum(1) - 1).max(), 0)
        self.assertListEqual(X1.tolist(), Z1.tolist())

    def test_whitening1(self):
        a = 2 * np.pi * np.arange(50) / 50
        X = np.stack([2 + np.cos(a), -1 + np.sin(a)], 1)
        X1 = pvml.whitening(X)
        self.assertAlmostEqual(X1.mean(0)[0], 0)
        self.assertAlmostEqual(X1.mean(0)[1], 0)
        c = np.cov(X1.T)
        self.assertAlmostEqual(c[0, 0], 1)
        self.assertAlmostEqual(c[0, 1], 0)
        self.assertAlmostEqual(c[1, 0], 0)
        self.assertAlmostEqual(c[1, 1], 1)

    def test_whitening2(self):
        X = (np.arange(30) % 4).reshape(10, 3)
        T = np.ones((5, 3))
        X1, T1 = pvml.whitening(X, T)
        self.assertAlmostEqual(np.abs(X1.mean(0)).max(), 0)
        c = np.cov(X1.T)
        self.assertAlmostEqual(np.abs(c - np.eye(3)).max(), 0)
        self.assertEqual(T1.std(0)[0], 0)
        self.assertEqual(T1.std(0)[1], 0)
        self.assertEqual(T1.std(0)[2], 0)

    def test_wrong_dimensions1(self):
        X = np.linspace(0, 1, 10)
        with self.assertRaises(ValueError):
            pvml.meanvar_normalization(X)

    def test_wrong_dimensions2(self):
        X1 = np.linspace(0, 1, 10).reshape(5, 2)
        X2 = np.linspace(0, 1, 12).reshape(4, 3)
        with self.assertRaises(ValueError):
            pvml.meanvar_normalization(X1, X2)


if __name__ == '__main__':
    unittest.main()
