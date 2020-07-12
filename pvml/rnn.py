import numpy as np
from .checks import _check_size
from .mlp import relu
from .multinomial_logistic import softmax, cross_entropy


# TODO:
# - Docstrings
# - Checks


class RNN:
    """Recurrent Neural Network."""
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = RNNBasicCell(input_size, hidden_size)
        self.W = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b = np.zeros(output_size)
        self.up_W = np.zeros_like(self.W)
        self.up_b = np.zeros_like(self.b)
        self.up_p = [np.zeros_like(p) for p in self.cell.parameters()]

    def forward(self, X):
        init = np.zeros((X.shape[0], self.W.shape[0]))
        H = self.cell.forward(X, init)
        P = self.activation(H @ self.W + self.b)
        return H, P

    def backward(self, H, P, Y):
        DU = self.activation_backward(P, Y)
        DH = DU @ self.W.T
        DZ = self.cell.backward(H, DH, np.zeros((H.shape[0], self.W.shape[0])))
        return DZ, DU

    def backpropagation(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99):
        H, P = self.forward(X)
        DZ, DU = self.backward(H, P, Y)
        h, k = self.W.shape
        grad_W = H.reshape(-1, h).T @ DU.reshape(-1, k) + lambda_ * self.W
        grad_b = DU.sum((0, 1))
        grad_p = self.cell.parameters_grad(X, H, DZ, np.zeros((X.shape[0], h)))
        self.up_W *= momentum
        self.up_W -= lr * grad_W
        self.W += self.up_W
        self.up_b *= momentum
        self.up_b -= lr * grad_b
        self.b += self.up_b
        for u, p, g in zip(self.up_p, self.cell.parameters(), grad_p):
            u *= momentum
            u -= lr * g
            p += u

    def activation(self, U):
        m, t, n = U.shape
        return softmax(U.reshape(-1, n)).reshape(m, t, n)

    def activation_backward(self, P, Y):
        if Y.ndim == 1:
            D = np.zeros_like(P)
            D[:, -1, :] = P[:, -1, :]
            # Implicitly subtract the one-hot vectors
            D[np.arange(Y.size), -1, Y] -= 1
        else:
            D = P.reshape(Y.size, -1).copy()
            # Implicitly subtract the one-hot vectors
            D[np.arange(Y.size), Y] -= 1
            D = D.reshape(*P.shape)
        return D / Y.shape[0]


class RNNBasicCell:
    """A Basic RNN cell."""
    #
    # Forward pass, given X:
    #   Z[t] = W X[t] + U H[t - 1] + b
    #   H[t] = a(Z[t])
    #   H[t] -> L[t]
    #   loss = sum(L[t])
    #
    # Backward pass, given DL[t] = dL[t] / dH[t]:
    #   DH[t] = DL[t] + U^T DZ[t + 1]
    #   DZ[t] = DH[t] * a'(Z[t])
    #
    def __init__(self, input_size, hidden_size):
        """Initialize the cell.

        Parameters
        ----------
        input_size : int
            number of input neurons.
        hidden_size : int
            number of hidden neurons.
        """
        self.W = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.U = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b = np.zeros(hidden_size)

    def forward(self, X, Hinit):
        """Forward step: return the hidden state at each time step.

        Parameters
        ----------
        X : ndarray, shape (m, t, n)
            sequence of input features (m sequences, t time steps, n components).
        Hinit : ndarray, shape (m, h)
            intial state (one vector for each input sequence).

        Returns
        -------
        H : ndarray, shape (m, t, h)
            sequence of hidden states (m sequences, t time steps, h units).
        """
        m, t, n = X.shape
        k = self.b.size
        H = (X.reshape(-1, n) @ self.W).reshape(m, t, k) + self.b
        for i in range(0, t):
            Z = H[:, i, :] + Hinit @ self.U
            H[:, i, :] = self.forward_activation(Z)
            Hinit = H[:, i, :]
        return H

    def backward(self, H, DL, DZinit):
        m, t, n = H.shape
        DZ = np.empty_like(H)
        DA = self.backward_activation(H, DL)
        for i in range(t - 1, -1, -1):
            DH = DL[:, i, :] + DZinit @ self.U.T
            DZinit = DA[:, i, :] * DH
            DZ[:, i, :] = DZinit
        return DZ

    def forward_activation(self, X):
        return relu(X)

    def backward_activation(self, X, D):
        return (X > 0) * D

    def parameters(self):
        return (self.W, self.U, self.b)

    def parameters_grad(self, X, H, DZ, DZinit):
        n, h = self.W.shape
        DU = H[:, :-1, :].reshape(-1, h).T @ DZ[:, 1:, :].reshape(-1, h)
        DU += H[:, -1, :].T @ DZinit
        Db = DZ.sum((0, 1))
        DW = X.reshape(-1, n).T @ DZ.reshape(-1, h)
        return (DW, DU, Db)


def _test():
    import matplotlib.pyplot as plt
    plt.ion()
    m = 100
    t = 5
    X = np.random.randint(0, 2, (m, t, 1))
    C = X.sum(1).sum(-1)
    C1 = C + np.random.uniform(-0.1, 0.1, *C.shape)
    Y = (C > (t // 2)).astype(int)
    rnn = RNN(1, 20, 2)
    for iter in range(1000000):
        rnn.backpropagation(X, Y, lr=0.0001)
        if iter % 1000 == 0:
            H, P = rnn.forward(X)
            Z = P[:, -1, :].argmax(-1)
            print(iter, (Z == Y).mean())
            plt.figure(0)
            plt.clf()
            plt.plot([0, t], [0.5, 0.5], 'k--')
            plt.scatter(C1, P[:, -1, 1], c=Y)
            DZ, DU = rnn.backward(H, P, Y)
            plt.figure(1)
            plt.clf()
            plt.plot((DU ** 2).mean(0).mean(-1))
            plt.title("DU")
            plt.figure(2)
            plt.clf()
            plt.plot((DZ ** 2).mean(0).mean(-1))
            plt.title("DZ")
            plt.pause(0.05)

_test()
