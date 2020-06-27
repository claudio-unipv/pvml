import numpy as np
import pvml
import unittest
import test_data


class TestSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train1(self):
        X, Y = test_data.separable_circle_data_set(50, 2)
        w, b = pvml.svm_train(X, Y, 0, lr=1e-1, steps=1000)
        Yhat, P = pvml.svm_inference(X, w, b)
        self.assertListEqual(Y.tolist(), Yhat.tolist())
        self.assertListEqual(Yhat.tolist(), (P > 0).tolist())

    def test_train2(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        w, b = pvml.svm_train(X, Y, 0.0001, lr=10, steps=1000)
        Yhat, P = pvml.svm_inference(X, w, b)
        self.assertListEqual(Y.tolist(), Yhat.tolist())
        self.assertListEqual(Yhat.tolist(), (P > 0).tolist())

    def test_train3(self):
        X, Y = test_data.separable_stripes_data_set(50, 2)
        w, b = pvml.svm_train(X, Y, 0.0001, lr=10, steps=1000)
        Yhat, P = pvml.svm_inference(X, w, b)
        self.assertListEqual(Y.tolist(), Yhat.tolist())
        self.assertListEqual(Yhat.tolist(), (P > 0).tolist())

    def test_hinge_loss1(self):
        Y = np.array([0, 1])
        Z = np.array([-1, 1])
        loss = pvml.hinge_loss(Y, Z)
        self.assertEqual(loss, 0)

    def test_hinge_loss2(self):
        Y = np.array([0, 1])
        Z = np.array([0, 0])
        loss = pvml.hinge_loss(Y, Z)
        self.assertEqual(loss, 1)

    def test_hinge_loss3(self):
        Y = np.array([0, 1])
        Z = np.array([1, -1])
        loss = pvml.hinge_loss(Y, Z)
        self.assertEqual(loss, 2)


if __name__ == '__main__':
    unittest.main()
