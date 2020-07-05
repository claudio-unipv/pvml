import numpy as np
import pvml
import unittest
import test_data


class TestMulticlassKSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train_ovo(self):
        for k in range(2, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                a, b = pvml.one_vs_one_ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
                Yhat = pvml.one_vs_one_ksvm_inference(X, X, a, b, "rbf", 0.1)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train_ovr(self):
        for k in range(2, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                a, b = pvml.one_vs_rest_ksvm_train(X, Y, "rbf", 0.1, 0, lr=1e-1, steps=1000)
                Yhat = pvml.one_vs_rest_ksvm_inference(X, X, a, b, "rbf", 0.1)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())


if __name__ == '__main__':
    unittest.main()
