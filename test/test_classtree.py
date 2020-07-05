import numpy as np
import pvml
import unittest
import test_data


class TestClassificationTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_train_no_pruning(self):
        tree = pvml.ClassificationTree()
        div = ["gini", "entropy", "error"]
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.separable_circle_data_set(50, k)
                tree.train(X, Y, diversity=div[k % 3], pruning_cv=0)
                Yhat = tree.inference(X)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_train_pruning(self):
        tree = pvml.ClassificationTree()
        div = ["gini", "entropy", "error"]
        for k in range(1, 6):
            with self.subTest(k):
                X, Y = test_data.separable_hypercubes_data_set(51, k)
                tree.train(X, Y, diversity=div[k % 3], pruning_cv=5)
                Yhat = tree.inference(X)[0]
                self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_dump(self):
        tree = pvml.ClassificationTree()
        X, Y = test_data.separable_hypercubes_data_set(21, 3)
        tree.train(X, Y)
        s = tree._dumps()
        self.assertTrue(len(s.splitlines()) > 3)

    def test_check1(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        Y = np.arange(5)
        tree = pvml.ClassificationTree()
        tree.train(X, Y, pruning_cv=0)
        with self.assertRaises(ValueError):
            tree.inference(np.arange(5))

    def test_check2(self):
        X = np.linspace(-1, 1, 10).reshape(5, 2)
        X[:, 0] = 0
        Y = np.arange(5)
        tree = pvml.ClassificationTree()
        tree.train(X, Y, pruning_cv=0)
        with self.assertRaises(ValueError):
            X = np.linspace(-1, 1, 5).reshape(5, 1)
            tree.inference(X)


if __name__ == '__main__':
    unittest.main()
