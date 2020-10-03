import numpy as np
import pvml
import io
import unittest
import test_data


class CNNTanh(pvml.CNN):
    """CNN with hyperbolic tangent activation function.

    It is used here because tanh is more reliable than ReLU for
    numerical testing.
    """
    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return np.tanh(X)

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        # y = tanh(x)  ==>  dy/dx = (1 - tanh(x)^2) = (1 - y^2)
        return d * (1 - Y ** 2)


class TestCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7477)

    def test_activation_gradient(self):
        """Test the gradient of the activation function."""
        cnn = CNNTanh([1, 1])
        X = np.random.randn(10, 1)
        Y = cnn.forward_hidden_activation(X)
        eps = 1e-7
        Y1 = cnn.forward_hidden_activation(X + eps)
        D = cnn.backward_hidden_activation(Y, np.ones_like(Y))
        D1 = (Y1 - Y) / eps
        error = np.abs(D1 - D).max()
        self.assertAlmostEqual(error, 0.0, 5)

    def test_padding(self):
        channels = [1, 1, 1, 1]
        kernel_sz = [3, 3, 3]
        strides = [1, 1, 1]
        pads = [0, 1, 2]
        cnn = pvml.CNN(channels, kernel_sz, strides, pads)
        X = np.random.randn(1, 11, 11, 1)
        A = cnn.forward(X)
        self.assertEqual(A[0].shape[1], 11)
        self.assertEqual(A[1].shape[1], 9)
        self.assertEqual(A[2].shape[1], 9)
        self.assertEqual(A[3].shape[1], 11)
        Y = np.zeros(X.shape[0], dtype=int)
        D = cnn.backward(Y, A)
        self.assertEqual(A[1].shape[1], D[0].shape[1])
        self.assertEqual(A[2].shape[1], D[1].shape[1])
        self.assertEqual(A[3].shape[1], D[2].shape[1])

    def test_backward(self):
        """Test the gradient of the loss wrt the input."""
        m = 3
        h, w = 7, 9
        n = 8
        k = 2
        channels = [n, 4, 4, k]
        kernel_sz = [3, 3, 3]
        strides = [1, 2, 1]
        pads = [0, 1, 2]

        net = CNNTanh(channels, kernel_sz, strides, pads)
        X = np.random.randn(m, h, w, n)
        Y = np.arange(m) % k
        A = net.forward(X)
        D = net.backward(Y, A)
        grad = net.backward_to_input(D[0], X.shape[1], X.shape[2])
        loss = net.loss(Y, A[-1])
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

    def test_parameter_gradients(self):
        """Test the gradient of the loss wrt the parameters."""
        m = 2
        h, w = 14, 18
        n = 3
        k = 5
        channels = [n, 4, 4, k]
        kernel_sz = [5, 3, 3]
        strides = [2, 1, 2]
        pads = [2, 1, 0]

        net = CNNTanh(channels, kernel_sz, strides, pads)
        X = np.random.randn(m, h, w, n)
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

    def test_train(self):
        X, Y = test_data.separable_hypercubes_data_set(50, 2)
        net = pvml.CNN([X.shape[1], 2], [1])
        X = X.reshape(X.shape[0], 1, 1, X.shape[1])
        net.train(X, Y, lr=1e-1, steps=1000)
        Yhat, P = net.inference(X)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_batch(self):
        X, Y = test_data.separable_circle_data_set(10, 2)
        net = pvml.CNN([X.shape[1], 2, 2], [1, 1])
        X = X.reshape(X.shape[0], 1, 1, X.shape[1])
        net.train(X, Y, lr=1e-1, steps=1000, batch=10)
        Yhat, P = net.inference(X)
        self.assertListEqual(Y.tolist(), Yhat.tolist())

    def test_saveload(self):
        channels = [2, 3, 4, 5]
        net1 = pvml.CNN(channels)
        f = io.BytesIO()
        net1.save(f)
        f.seek(0)
        net2 = pvml.CNN.load(f)
        for w1, w2 in zip(net1.weights, net2.weights):
            self.assertListEqual(list(w1.ravel()), list(w2.ravel()))
        for b1, b2 in zip(net1.biases, net2.biases):
            self.assertListEqual(list(b1), list(b2))


if __name__ == '__main__':
    unittest.main()
