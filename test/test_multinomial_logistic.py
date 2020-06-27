import numpy as np
import pvml
import unittest
import test_data


class TestMultinomialLogistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train1(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                w, b = pvml.multinomial_logreg_train(X, Y, 0, lr=1e-1, steps=1000)
                P = pvml.multinomial_logreg_inference(X, w, b)
                Yhat = P.argmax(1)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train2(self):
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.separable_hypercubes_data_set(51, k)
                w, b = pvml.multinomial_logreg_train(X, Y, 1e-4, lr=1, steps=1000)
                P = pvml.multinomial_logreg_inference(X, w, b)
                Yhat = P.argmax(1)
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_softmax(self):
        Z = np.random.randn(20, 4)
        Y = np.argmax(Z, 1)
        P = pvml.softmax(Z)
        Yhat = P.argmax(1)
        error = np.abs(P.sum(1) - 1).max()
        self.assertAlmostEqual(0, error)

    def test_one_hot_vectors(self):
        k = 7
        Y = np.arange(20) % k
        H = pvml.one_hot_vectors(Y, k)
        self.assertListEqual([1] * 20, H.sum(1).tolist())
        self.assertListEqual(np.bincount(Y, minlength=k).tolist() , H.sum(0).tolist())
        
    def test_cross_entropy(self):
        P = np.array([[1, 0, 0]])
        H = np.array([[1, 0, 0]])
        olderr = np.seterr(all='ignore')
        ce = pvml.cross_entropy(H, P)
        np.seterr(**olderr)
        self.assertEqual(0, ce)

        
if __name__ == '__main__':
    unittest.main()
