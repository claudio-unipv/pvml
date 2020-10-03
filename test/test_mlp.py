import numpy as np
import pvml
import io
import unittest
import test_data


class MLPTanh(pvml.MLP):
    """MLP with hyperbolic tangent activation function.

    It is used here because tanh is more reliable than ReLU for
    numerical testing.
    """
    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return np.tanh(X)

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        return d * (1 - Y ** 2)


class TestMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_parameter_gradients(self):
        """Test the gradient of the loss wrt the parameters."""
        m = 4
        n = 3
        k = 5
        net = MLPTanh([n, 2, 7, k])
        X = np.random.randn(m, n)
        Y = np.arange(m) % k
        A = net.forward(X)
        D = net.backward(Y, A)
        loss = net.loss(Y, A[-1])
        grad_Ws, grad_bs = net.parameters_gradient(A, D)
        eps = 1e-7
        for ps, grad_ps, name in ((net.weights, grad_Ws, "W"), (net.biases, grad_bs, "b")):
            for p, grad_p, l in zip(ps, grad_ps, range(len(ps))):
                for index in np.ndindex(*p.shape):
                    backup = p[index]
                    p[index] += eps
                    with self.subTest(parameter=f"{name}[{l}]", index=index):
                        A1 = net.forward(X)
                        loss1 = net.loss(Y, A1[-1])
                        ratio = (loss1 - loss) / eps
                        self.assertAlmostEqual(grad_p[index], ratio, 5)
                    p[index] = backup

    def test_backward(self):
        m = 3
        n = 2
        k = 3
        net = MLPTanh([n, 2, 3, 3, 2, k])
        X = np.random.randn(m, n)
        Y = np.arange(m) % k
        A = net.forward(X)
        loss = net.loss(Y, A[-1])
        D = net.backward(Y, A)
        grad = net.backward_to_input(D[0])
        eps = 1e-7
        for index in np.ndindex(*X.shape):
            backup = X[index]
            X[index] += eps
            with self.subTest(index=index):
                A1 = net.forward(X)
                loss1 = net.loss(Y, A1[-1])
                ratio = (loss1 - loss) / eps
                self.assertAlmostEqual(grad[index], ratio, 5)
            X[index] = backup

    def test_train(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        net = pvml.MLP([X.shape[1], 2])
        net.train(X, Y, lr=1e-1, steps=1000)
        Yhat, P = net.inference(X)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_batch(self):
        X, Y = test_data.separable_circle_data_set(50, 2)
        net = pvml.MLP([X.shape[1], 2, 2])
        net.train(X, Y, lr=1e-1, steps=1000, batch=10)
        Yhat, P = net.inference(X)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_saveload(self):
        neurons = [2, 3, 4, 5]
        net1 = pvml.MLP(neurons)
        f = io.BytesIO()
        net1.save(f)
        f.seek(0)
        net2 = pvml.MLP.load(f)
        for w1, w2 in zip(net1.weights, net2.weights):
            self.assertListEqual(list(w1.ravel()), list(w2.ravel()))
        for b1, b2 in zip(net1.biases, net2.biases):
            self.assertListEqual(list(b1), list(b2))


if __name__ == '__main__':
    unittest.main()
