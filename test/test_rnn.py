import numpy as np
import pvml
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
        DZ = cell.backward(H, np.ones_like(H), init)
        gradients = cell.parameters_grad(X, H, DZ, init)
        eps = 1e-7
        for p in range(3):
            for index in np.ndindex(*gradients[p].shape):
                backup = cell.parameters()[p][index]
                with self.subTest(parameter=p, index=index):
                    cell.parameters()[p][index] += eps
                    H1 = cell.forward(X, init)
                    L1 = H1.sum()
                    D = (L1 - L) / eps
                    self.assertAlmostEqual(gradients[p][index], D, 5)
                cell.parameters()[p][index] = backup


if __name__ == '__main__':
    unittest.main()
