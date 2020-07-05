import numpy as np
import pvml
import unittest
import test_data


class TestMulticlassSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train_ovo(self):
        for k in range(2, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                w, b = pvml.one_vs_one_svm_train(X, Y, 0, lr=1e-1, steps=1000)
                Yhat = pvml.one_vs_one_svm_inference(X, w, b)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train_ovr(self):
        for k in range(2, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                w, b = pvml.one_vs_rest_svm_train(X, Y, 0, lr=1e-1, steps=1000)
                Yhat = pvml.one_vs_rest_svm_inference(X, w, b)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())


if __name__ == '__main__':
    unittest.main()
