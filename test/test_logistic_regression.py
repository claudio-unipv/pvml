import numpy as np
import pvml
import unittest
import test_data


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train_l2(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        w, b = pvml.logreg_train(X, Y, 0.0001, lr=10, steps=1000)
        P = pvml.logreg_inference(X, w, b)
        Yhat = (P > 0.5).astype(int)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train_l1(self):
        X, Y = test_data.separable_stripes_data_set(50, 2)
        w, b = pvml.logreg_l1_train(X, Y, 0.0001, lr=10, steps=1000)
        P = pvml.logreg_inference(X, w, b)
        Yhat = (P > 0.5).astype(int)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_cross_entropy1(self):
        Y = np.array([0, 0, 1, 1])
        e = pvml.binary_cross_entropy(Y, Y)
        self.assertEqual(e, 0)

    def test_cross_entropy2(self):
        Y = np.array([0, 0, 1, 1])
        e = pvml.binary_cross_entropy(Y, 1 - Y)
        self.assertTrue(np.isinf(e))


if __name__ == '__main__':
    unittest.main()
