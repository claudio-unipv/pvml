import numpy as np
import pvml
import unittest
import test_data


class TestGDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_heteroscedastic_gda(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(48, k)
                means, icovs, priors = pvml.hgda_train(X, Y)
                Yhat, scores = pvml.hgda_inference(X, means, icovs, priors)
                self.assertListEqual(Y.tolist(), Yhat.tolist())
                self.assertListEqual(priors.tolist(), [1.0 / k] * k)

    def test_heteroscedastic_priors(self):
        k = 3
        X, Y = test_data.non_separable_checkerboard_data_set(24, k)
        means, icovs, priors = pvml.hgda_train(X, Y, priors=np.array([1, 0, 0]))
        Yhat, scores = pvml.hgda_inference(X, means, icovs, priors)
        self.assertListEqual([0] * 24, Yhat.tolist())

    def test_omoscedastic_gda(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.separable_stripes_data_set(48, k)
                w, b = pvml.ogda_train(X, Y)
                Yhat, scores = pvml.ogda_inference(X, w, b)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_omoscedastic_priors(self):
        k = 4
        X, Y = test_data.non_separable_checkerboard_data_set(24, k)
        w, b = pvml.ogda_train(X, Y, priors=np.array([1, 0, 0, 0]))
        Yhat, scores = pvml.ogda_inference(X, w, b)
        self.assertListEqual([0] * 24, Yhat.tolist())

    def test_mindist(self):
        for k in range(1, 5):
            with self.subTest(k):
                X, Y = test_data.separable_hypercubes_data_set(48, k)
                means = pvml.mindist_train(X, Y)
                Yhat, scores = pvml.mindist_inference(X, means)
                self.assertListEqual(Y.tolist(), Yhat.tolist())


if __name__ == '__main__':
    unittest.main()
