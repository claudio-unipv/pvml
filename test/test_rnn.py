import numpy as np
import pvml
import io
import unittest


class TestRNNBaseCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_gradientW(self):
        """Test the gradient of the loss wrt the parameters.

        The loss here is just the sum of all the cell states.
        """
        n, h, m, t = 3, 2, 4, 5
        init = np.zeros((m, h))
        cell = pvml.RNNBasicCell(n, h)
        X = np.random.randn(m, t, n)
        H = cell.forward(X, np.zeros((m, h)))
        L = H.sum()
        cell.backward(np.ones_like(H))
        gradients = cell.parameters_grad()
        eps = 1e-7
        for p in range(len(gradients)):
            for index in np.ndindex(*gradients[p].shape):
                backup = cell.parameters()[p][index]
                with self.subTest(parameter=p, index=index):
                    cell.parameters()[p][index] += eps
                    H1 = cell.forward(X, init)
                    L1 = H1.sum()
                    D = (L1 - L) / eps
                    self.assertAlmostEqual(gradients[p][index], D, 5)
                cell.parameters()[p][index] = backup


class TestRNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_gradient_single_label(self):
        neurons = [3, 2, 4, 2, 3]
        m, t = 3, 5
        rnn = pvml.RNN(neurons)
        X = np.random.randn(m, t, neurons[0])
        Y = np.arange(m) % neurons[-1]
        P = rnn.forward(X)
        L = rnn.loss(Y, P)
        DX = rnn.backward(Y)
        eps = 1e-7
        for index in np.ndindex(*X.shape):
            backup = X[index]
            with self.subTest(index=index):
                X[index] += eps
                P1 = rnn.forward(X)
                L1 = rnn.loss(Y, P1)
                D = (L1 - L) / eps
                self.assertAlmostEqual(DX[index], D, 5)
            X[index] = backup

    def test_gradient_sequence(self):
        neurons = [3, 2, 4, 2, 3]
        m, t = 3, 5
        rnn = pvml.RNN(neurons)
        X = np.random.randn(m, t, neurons[0])
        Y = np.arange(m * t).reshape(m, t) % neurons[-1]
        P = rnn.forward(X)
        L = rnn.loss(Y, P)
        DX = rnn.backward(Y)
        eps = 1e-7
        for index in np.ndindex(*X.shape):
            backup = X[index]
            with self.subTest(index=index):
                X[index] += eps
                P1 = rnn.forward(X)
                L1 = rnn.loss(Y, P1)
                D = (L1 - L) / eps
                self.assertAlmostEqual(DX[index], D, 5)
            X[index] = backup

    def test_train(self):
        k = 4
        neurons = [k, 2 * k, k]
        # Train to echo output a set sequence
        Y = np.arange(k).reshape(1, k)
        X = np.eye(k).reshape(1, k, k)
        rnn = pvml.RNN(neurons)
        rnn.train(X, Y, lr=0.01, steps=100)
        P = rnn.forward(X)
        Z = P[0, :, :].argmax(-1)
        self.assertListEqual(list(Z), list(Y[0, :]))

    def test_saveload(self):
        neurons = [2, 3, 4, 5]
        rnn = pvml.RNN(neurons)
        f = io.BytesIO()
        rnn.save(f)
        f.seek(0)
        rnn2 = pvml.RNN.load(f)
        self.assertListEqual(list(rnn.b), list(rnn2.b))


class TestGRUCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_forward(self):
        n, h, m, t = 5, 3, 2, 4
        init = np.zeros((m, h))
        cell = pvml.GRUCell(n, h)
        X = np.random.randn(m, t, n)
        H = cell.forward(X, init)
        self.assertEqual(H.shape, (m, t, h))

    def test_gradientX(self):
        n, h, m, t = 2, 2, 2, 3
        init = np.zeros((m, h))
        cell = pvml.GRUCell(n, h)
        X = np.random.randn(m, t, n)
        H = cell.forward(X, init)
        DL = np.ones_like(H)
        DX = cell.backward(DL)
        L0 = H.sum()
        eps = 1e-7
        for index in np.ndindex(*X.shape):
            backup = X[index]
            X[index] += eps
            with self.subTest(index=index):
                H1 = cell.forward(X, init)
                L1 = H1.sum()
                D = (L1 - L0) / eps
                self.assertAlmostEqual(DX[index], D, 5)
            X[index] = backup

    def test_gradientW(self):
        """Test the gradient of the loss wrt the parameters.

        The loss here is just the sum of all the cell states.
        """
        n, h, m, t = 3, 2, 4, 5
        init = np.zeros((m, h))
        cell = pvml.GRUCell(n, h)
        X = np.random.randn(m, t, n)
        H = cell.forward(X, np.zeros((m, h)))
        L = H.sum()
        cell.backward(np.ones_like(H))
        gradients = cell.parameters_grad()
        eps = 1e-7
        for p in range(len(gradients)):
            for index in np.ndindex(*gradients[p].shape):
                backup = cell.parameters()[p][index]
                with self.subTest(parameter=p, index=index):
                    cell.parameters()[p][index] += eps
                    H1 = cell.forward(X, init)
                    L1 = H1.sum()
                    D = (L1 - L) / eps
                    self.assertAlmostEqual(gradients[p][index], D, 5)
                cell.parameters()[p][index] = backup


class TestRNNAbstractCell(unittest.TestCase):
    def test_forward(self):
        n, h, m, t = 3, 2, 4, 5
        init = np.zeros((m, h))
        cell = pvml.RNNAbstractCell()
        X = np.zeros((m, t, n))
        with self.assertRaises(NotImplementedError):
            cell.forward(X, init)

    def test_backward(self):
        n, m, t = 3, 4, 5
        D = np.zeros((m, t, n))
        cell = pvml.RNNAbstractCell()
        with self.assertRaises(NotImplementedError):
            cell.backward(D)

    def test_parameters(self):
        cell = pvml.RNNAbstractCell()
        with self.assertRaises(NotImplementedError):
            cell.parameters()

    def test_parameters_grad(self):
        cell = pvml.RNNAbstractCell()
        with self.assertRaises(NotImplementedError):
            cell.parameters_grad()


if __name__ == '__main__':
    unittest.main()
