import numpy as np
import pvml
import unittest
import test_data


class TestPerceptron(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train1(self):
        X, Y = test_data.separable_circle_data_set(50, 2)
        w, b = pvml.perceptron_train(X, Y)
        Yhat, Z = pvml.perceptron_inference(X, w, b)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train2(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        w, b = pvml.perceptron_train(X, Y)
        Yhat, Z = pvml.perceptron_inference(X, w, b)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train3(self):
        X, Y = test_data.non_separable_checkerboard_data_set(51, 2)
        w, b = pvml.perceptron_train(X, Y, steps=100)
        Yhat, Z = pvml.perceptron_inference(X, w, b)
        self.assertGreater((Yhat == Y).sum(), 25)


if __name__ == '__main__':
    unittest.main()
