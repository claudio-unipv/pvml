import numpy as np
import pvml
import unittest
import test_data


class TestKSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train1(self):
        X, Y = test_data.separable_circle_data_set(50, 2)
        alpha, b = pvml.ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
        Yhat, P = pvml.ksvm_inference(X, X, alpha, b, "rbf", 0.1)
        self.assertListEqual(Y.tolist(), Yhat.tolist())
        self.assertListEqual(Yhat.tolist(), (P > 0).tolist())

    def test_train2(self):
        X, Y = test_data.separable_hypercubes_data_set(12, 2)
        alpha, b = pvml.ksvm_train(X, Y, "polynomial", 2, 0, lr=10, steps=1000)
        Yhat, P = pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)
        self.assertListEqual(Y.tolist(), Yhat.tolist())
        self.assertListEqual(Yhat.tolist(), (P > 0).tolist())

    def test_rbf_kernel(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        K = pvml.kernel(X, X, "rbf", 1)
        self.assertListEqual(np.diag(K).tolist(), [1] * 5)
        evals = np.linalg.eigvalsh(K)
        self.assertGreaterEqual(evals.min(), 0)

    def test_polynomial_kernel(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        K = pvml.kernel(X, X, "polynomial", 2)
        evals = np.linalg.eigvalsh(K)
        self.assertGreaterEqual(evals.min(), 0)

    def test_unknown_kernel(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        with self.assertRaises(ValueError):
            pvml.kernel(X, X, "unknown", 2)

    def test_wrong_dimensions1(self):
        X1 = np.linspace(-1, 1, 10).reshape(5, 2)
        X2 = np.linspace(-1, 1, 12).reshape(4, 3)
        with self.assertRaises(ValueError):
            pvml.kernel(X1, X2, "polynomial", 2)

    def test_wrong_dimensions2(self):
        X1 = np.linspace(-1, 1, 10)
        X2 = np.linspace(-1, 1, 10)
        with self.assertRaises(ValueError):
            pvml.kernel(X1, X2, "polynomial", 2)

    def test_wrong_coefficients1(self):
        X = np.linspace(-1, 1, 10)
        alpha = np.linspace(-1, 1, 10)
        b = 0
        with self.assertRaises(ValueError):
            pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)

    def test_wrong_coefficients2(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        alpha = np.linspace(-1, 1, 10)
        b = 0
        with self.assertRaises(ValueError):
            pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)

    def test_wrong_coefficients3(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        alpha = np.linspace(-1, 1, 10).reshape(5, 2)
        b = 0
        with self.assertRaises(ValueError):
            pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)

    def test_wrong_bias(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        alpha = np.linspace(-1, 1, 5)
        b = np.linspace(-1, 1, 5)
        with self.assertRaises(ValueError):
            pvml.ksvm_inference(X, X, alpha, b, "polynomial", 2)


if __name__ == '__main__':
    unittest.main()
