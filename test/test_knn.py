import numpy as np
import pvml
import unittest
import test_data


class TestKNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_knn_1(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.non_separable_checkerboard_data_set(48, k)
                Yhat, P = pvml.knn_inference(X, X, Y, k=1)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_knn_5(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(48, k)
                Yhat, P = pvml.knn_inference(X, X, Y, k=5)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_knn_auto_k(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(24, k)
                k, _ = pvml.knn_select_k(X, Y, maxk=7)
                self.assertGreater(k, 0)
                self.assertLess(k, 8)


if __name__ == '__main__':
    unittest.main()
