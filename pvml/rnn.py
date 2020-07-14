import numpy as np
from .checks import _check_size, _check_labels
from .mlp import relu
from .multinomial_logistic import softmax, cross_entropy


# TODO:
# - Docstrings
# - multilayer
# - tests
# - load/save
# - LSTM (peephole version?)


class RNN:
    """Recurrent Neural Network."""
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = RNNBasicCell(input_size, hidden_size)
        self.W = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b = np.zeros(output_size)
        self.reset_momentum()

    def reset_momentum(self):
        self.up_W = np.zeros_like(self.W)
        self.up_b = np.zeros_like(self.b)
        self.up_p = [np.zeros_like(p) for p in self.cell.parameters()]

    def forward(self, X):
        init = np.zeros((X.shape[0], self.W.shape[0]))
        H = self.cell.forward(X, init)
        V = H @ self.W + self.b
        P = self.activation(V)
        return H, P

    def backward(self, H, P, Y):
        DV = self.activation_backward(P, Y)
        DH = DV @ self.W.T
        DZ = self.cell.backward(H, DH, np.zeros((H.shape[0], self.W.shape[0])))
        return DZ, DV

    def train(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99,
              steps=10000, batch=None):
        """Train the network.

        Apply multiple steps of backpropagation to train the network.

        Parameters
        ----------
        X : ndarray, shape (m, t, n)
            input sequences (a batch of m sequences of length t and with n features).
        Y : ndarray, shape (m,) or (m, t)
            target output (integer class labels, one per sequence or one per sequence element).
        lr : float
            learning rate.
        lambda_ : float
            regularization coefficients.
        momentum : float
            momentum coefficient.
        steps : int
            training iterations.
        batch : int or None
            size of the minibatch used in each step.  When None all
            the data is used in each step.
        """
        if Y.ndim == 1:
            _check_size("mtn, m", X, Y)
        else:
            _check_size("mtn, mt", X, Y)
        Y = _check_labels(Y, self.W.shape[1])
        m = X.shape[0]
        if batch is None:
            batch = X.shape[0]
        i = m
        indices = np.arange(m)
        for step in range(steps):
            if i + batch > m:
                i = 0
                np.random.shuffle(indices)
            self.backpropagation(X[indices[i:i + batch], ...],
                                 Y[indices[i:i + batch], ...],
                                 lr=lr,
                                 lambda_=lambda_,
                                 momentum=momentum)
            i += batch
    
    def backpropagation(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99):
        H, P = self.forward(X)
        DZ, DV = self.backward(H, P, Y)
        h, k = self.W.shape
        grad_W = H.reshape(-1, h).T @ DV.reshape(-1, k) + lambda_ * self.W
        grad_b = DV.sum((0, 1))
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

    def save(self, filename):
        """Save the network to the file."""
        np.savez(filename, W=self.W, b=self.b, cells=self.cells)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        # !!!
        data = np.load(filename, allow_pickle=True)
        neurons = [w.shape[0] for w in data["weights"]]
        neurons.append(data["weights"][-1].shape[1])
        network = cls(neurons)
        network.weights = data["weights"]
        network.biases = data["biases"]
        return network
            
    def activation(self, V):
        m, t, n = V.shape
        return softmax(V.reshape(-1, n)).reshape(m, t, n)

    def activation_backward(self, P, Y):
        if Y.ndim == 1:
            D = np.zeros_like(P)
            D[:, -1, :] = P[:, -1, :]
            # Implicitly subtract the one-hot vectors
            D[np.arange(Y.size), -1, Y] -= 1
        else:
            D = P.reshape(Y.size, -1).copy()
            # Implicitly subtract the one-hot vectors
            D[np.arange(Y.size), Y.reshape(-1)] -= 1
            D = D.reshape(*P.shape)
        return D / Y.shape[0]

    def loss(self, Y, P):
        """Compute the average cross-entropy."""
        if Y.ndim == 1:
            return cross_entropy(Y, P[:, -1, :])
        else:
            return cross_entropy(Y.reshape(-1), P.reshape(Y.size, -1))


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
        _check_size("mtn, mh, nh", X, Hinit, self.W)
        m, t, n = X.shape
        X1 = (X.reshape(-1, n) @ self.W).reshape(m, t, -1) + self.b
        H = np.empty_like(X1)
        for i in range(0, t):
            Z = X1[:, i, :] + Hinit @ self.U
            H[:, i, :] = self.forward_activation(Z)
            Hinit = H[:, i, :]
        return H

    def backward(self, H, DL, DZinit):
        _check_size("mth, mth, mh", H, DL, DZinit)
        m, t, n = H.shape
        DZ = np.empty_like(H)
        for i in range(t - 1, -1, -1):
            DH = DL[:, i, :] + DZinit @ self.U.T
            DZinit = self.backward_activation(H[:, i, :]) * DH
            DZ[:, i, :] = DZinit
        return DZ

    def forward_activation(self, X):
        return relu(X)

    def backward_activation(self, H):
        return (H > 0).astype(float)

    def parameters(self):
        return (self.W, self.U, self.b)

    def parameters_grad(self, X, H, DZ, DZinit):
        n, h = self.W.shape
        DV = H[:, :-1, :].reshape(-1, h).T @ DZ[:, 1:, :].reshape(-1, h)
        DV += H[:, -1, :].T @ DZinit
        Db = DZ.sum((0, 1))
        DW = X.reshape(-1, n).T @ DZ.reshape(-1, h)
        return (DW, DV, Db)


def _test():
    # rnn = RNN(1, 1, 2)
    # rnn.cell.W[0, 0] = 2
    # rnn.cell.U[0, 0] = 3
    # rnn.W[0, 0] = 1
    # rnn.W[0, 1] = 0
    # X = np.ones((1, 5, 1))
    # Y = np.ones(1, dtype=int)
    # rnn.backpropagation(X, Y, lr=1)
    # return
    import matplotlib.pyplot as plt
    plt.ion()
    m = 100
    t = 5
    X = np.random.randint(0, 2, (m, t, 1))
    C = X.sum(1).sum(-1)
    C1 = C + np.random.uniform(-0.1, 0.1, *C.shape)
    Y = C
    rnn = RNN(1, 5, t + 1)
    for iter in range(1000000):
        i = iter % m
        rnn.backpropagation(X[i:i+1, ...], Y[i:i+1], lr=0.0001)
        if iter % 1000 == 0:
            H, P = rnn.forward(X)
            Z = P[:, -1, :].argmax(-1)
            print(iter, (Z == Y).mean())
            plt.figure(0)
            plt.clf()
            plt.plot([0, t], [0.5, 0.5], 'k--')
            plt.scatter(C1, Z, c=Y)
            DZ, DV = rnn.backward(H, P, Y)
            plt.figure(1)
            plt.clf()
            plt.plot((DV ** 2).mean(0).mean(-1))
            plt.title("DV")
            plt.figure(2)
            plt.clf()
            plt.plot((DZ ** 2).mean(0).mean(-1))
            plt.title("DZ")
            plt.pause(0.05)

def _test2():
    # rnn = RNN(1, 1, 2)
    # rnn.cell.W[0, 0] = 2
    # rnn.cell.U[0, 0] = 3
    # rnn.W[0, 0] = 1
    # rnn.W[0, 1] = 0
    # X = np.ones((1, 5, 1))
    # Y = np.ones(1, dtype=int)
    # rnn.backpropagation(X, Y, lr=1)
    # return
    import matplotlib.pyplot as plt
    plt.ion()
    m = 101
    t = 10
    h = 10
    delay = 3
    X = np.random.randint(0, 2, (m, t, 1))
    Y = np.zeros((m, t), dtype=int)
    Y[:, delay:] = X[:, :-delay, 0]
    rnn = RNN(1, 5, t + 1)
    for iter in range(0, 10000, 1000):
        rnn.train(X, Y, lr=0.0001, steps=1000, batch=1)
        H, P = rnn.forward(X)
        loss = rnn.loss(Y, P)
        Z = P.argmax(-1)
        print(iter, (Z == Y).mean(), loss)
        # plt.figure(0)
        # plt.clf()
        # plt.plot(X[i, :, 0])
        # plt.plot(Z[i, :])
        # DZ, DV = rnn.backward(H, P, Y)
        # plt.figure(1)
        # plt.clf()
        # plt.plot((DV ** 2).mean(0).mean(-1))
        # plt.title("DV")
        # plt.figure(2)
        # plt.clf()
        # plt.plot((DZ ** 2).mean(0).mean(-1))
        # plt.title("DZ")
        # plt.pause(0.05)

            
_test2()
