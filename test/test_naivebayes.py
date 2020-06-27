import numpy as np
import pvml
import unittest
import test_data


class TestNaiveBayes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_categorical(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.categorical_data_set(50, k)
                probs, priors = pvml.categorical_naive_bayes_train(X, Y)
                Yhat, scores = pvml.categorical_naive_bayes_inference(X, probs, priors)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_categorical2(self):
        X, Y = test_data.categorical_data_set(50, 2)
        priors = np.array([0.0, 1.0])
        probs, priors = pvml.categorical_naive_bayes_train(X, Y, priors)
        self.assertListEqual(priors.tolist(), [0.0, 1.0])
        Yhat, _ = pvml.categorical_naive_bayes_inference(X, probs, priors)
        self.assertListEqual([1] * Y.size, Yhat.tolist())

    def test_multinomial(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.bow_data_set(50, k)
                w, b = pvml.multinomial_naive_bayes_train(X, Y)
                Yhat, scores = pvml.multinomial_naive_bayes_inference(X, w, b)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_multinomial2(self):
        X, Y = test_data.bow_data_set(50, 2)
        priors = np.array([0.0, 1.0])
        w, b = pvml.multinomial_naive_bayes_train(X, Y, priors)
        Yhat, _ = pvml.multinomial_naive_bayes_inference(X, w, b)
        self.assertListEqual([1] * Y.size, Yhat.tolist())

    def test_gaussian(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.separable_hypercubes_data_set(30, k)
                ms, vs, ps = pvml.gaussian_naive_bayes_train(X, Y)
                Yhat, scores = pvml.gaussian_naive_bayes_inference(X, ms, vs, ps)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_gaussian2(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        priors = np.array([0.0, 1.0])
        ms, vs, priors = pvml.gaussian_naive_bayes_train(X, Y, priors)
        Yhat, _ = pvml.gaussian_naive_bayes_inference(X, ms, vs, priors)
        self.assertListEqual([1] * Y.size, Yhat.tolist())


if __name__ == '__main__':
    unittest.main()
