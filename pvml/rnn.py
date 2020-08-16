import numpy as np
from .checks import _check_size, _check_labels
from .mlp import relu
from .multinomial_logistic import softmax, cross_entropy
from .logistic_regression import sigmoid


# TODO:
# - Docstrings
# - LSTM (peephole version?)
# - GRU


class RNN:
    """Recurrent Neural Network."""
    def __init__(self, neuron_counts):
        self.neuron_counts = neuron_counts
        connections = zip(neuron_counts[:-2], neuron_counts[1:-1])
        self.cells = [RNNBasicCell(n1, n2) for n1, n2 in connections]
        self.W = (np.random.randn(neuron_counts[-2], neuron_counts[-1]) *
                  np.sqrt(2 / neuron_counts[-2]))
        self.b = np.zeros(neuron_counts[-1])
        self.reset_momentum()

    def reset_momentum(self):
        self.up_W = np.zeros_like(self.W)
        self.up_b = np.zeros_like(self.b)
        self.up_p = [[np.zeros_like(p) for p in c.parameters()] for c in self.cells]

    def forward(self, X, inits=None):
        m = X.shape[0]
        Hs = [X]
        if inits is None:
            inits = [np.zeros((m, h)) for h in self.neuron_counts[1:-1]]
        for cell, init in zip(self.cells, inits):
            X = cell.forward(X, init)
            Hs.append(X)
        V = X @ self.W + self.b
        P = self.activation(V)
        return Hs, P

    def backward(self, Hs, P, Y, inits=None):
        DV = self.activation_backward(P, Y)
        DH = DV @ self.W.T
        DZs = []
        if inits is None:
            inits = [np.zeros((P.shape[0], h)) for h in self.neuron_counts[1:-1]]
        for H, cell, init in zip(Hs[::-1], self.cells[::-1], inits[::-1]):
            DZ, DH = cell.backward(H, DH, init)
            DZs.append(DZ)
        return DZs[::-1], DH, DV

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
        _check_size(("mtn, m" if Y.ndim == 1 else "mtn, mt"), X, Y)
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
        Hs, P = self.forward(X)
        DZs, DX, DV = self.backward(Hs, P, Y)
        h, k = self.W.shape
        grad_W = Hs[-1].reshape(-1, h).T @ DV.reshape(-1, k) + lambda_ * self.W
        grad_b = DV.sum((0, 1))

        grads = []
        for cell, H, DZ, h in zip(self.cells, Hs[1:], DZs, self.neuron_counts[1:]):
            init = np.zeros((X.shape[0], h))
            g = cell.parameters_grad(X, H, DZ, init)
            X = H
            grads.append(g)

        self.up_W *= momentum
        self.up_W -= lr * grad_W
        self.W += self.up_W
        self.up_b *= momentum
        self.up_b -= lr * grad_b
        self.b += self.up_b
        for cell, grad, up in zip(self.cells, grads, self.up_p):
            for u, p, g in zip(up, cell.parameters(), grad):
                u *= momentum
                u -= lr * g
                p += u

    def save(self, filename):
        """Save the network to the file."""
        cell_params = [p for cell in self.cells for p in cell.parameters()]
        np.savez(filename, neuron_counts=self.neuron_counts, W=self.W, b=self.b, *cell_params)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        data = np.load(filename)
        network = cls(data["neuron_counts"])
        network.W[...] = data["W"]
        network.b[...] = data["b"]
        index = 0
        for cell in network.cells:
            for p in cell.parameters():
                p[...] = data["arr_" + str(index)]
                index += 1
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
        return D / Y.size

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
        # X1 = (X.reshape(-1, n) @ self.W).reshape(m, t, -1) + self.b
        X1 = X @ self.W + self.b
        H = np.empty_like(X1)
        for i in range(0, t):
            Z = X1[:, i, :] + Hinit @ self.U
            H[:, i, :] = self.forward_activation(Z)
            Hinit = H[:, i, :]
        return H

    def backward(self, H, DL, DZinit):
        """Backward step: return derivatives computed for all time steps.

        Parameters
        ----------
        H : ndarray, shape (m, t, h)
            sequence of states obtained in the forward step.
        DL : ndarray, shape (m, t, h)
            derivative of the losses at each time steps with respect to the corresponding states.
        DZinit : ndarray, shape (m, h)
            derivative of the total loss with respect to the last logits obtained in the forward
            step.

        Returns
        -------
        DZ : ndarray, shape (m, t, h)
            derivative of the losses at each time steps with respect to the logits.
        DX : ndarray, shape (m, t, n)
            derivative of the losses at each time steps with respect to the inputs.
        """

        _check_size("mth, mth, mh", H, DL, DZinit)
        m, t, n = H.shape
        DZ = np.empty_like(H)
        for i in range(t - 1, -1, -1):
            DH = DL[:, i, :] + DZinit @ self.U.T
            DZinit = self.backward_activation(H[:, i, :]) * DH
            DZ[:, i, :] = DZinit
        DX = DZ @ self.W.T
        return DZ, DX

    def forward_activation(self, X):
        """Activation function."""
        return relu(X)

    def backward_activation(self, A):
        """Derivative of the activation, given the activation values."""
        return (A > 0).astype(float)

    def parameters(self):
        """List of parameters of the cell."""
        return (self.W, self.U, self.b)

    def parameters_grad(self, X, H, DZ, DZinit):
        """Derivative of the total loss with respect to the parameters of the cell."""
        n, h = self.W.shape
        DV = H[:, :-1, :].reshape(-1, h).T @ DZ[:, 1:, :].reshape(-1, h)
        DV += H[:, -1, :].T @ DZinit
        Db = DZ.sum((0, 1))
        DW = X.reshape(-1, n).T @ DZ.reshape(-1, h)
        return (DW, DV, Db)

class GRUCell:
    """Gated Recurring Unit cell."""
    #
    # Forward pass, given X:
    #   Sz[t] = Wz X[t] + Uz H[t - 1] + bz
    #   Z[t] = sigmoid(Sz[t])
    #   Sr[t] = Wr X[t] + Ur H[t - 1] + br
    #   R[t] = sigmoid(Sr[t])
    #   Sh[t] = Wh X[t] + Uh (R[t] * H[t - 1]) + bh
    #   H[t] = (1 - Z[t]) * H[t - 1] + Z[t] * a(Sh[t])
    #   H[t] -> L[t]
    #   loss = sum(L[t])
    #
    # Backward pass, given DL[t] = dL[t] / dH[t]:
    #   !!! DH[t] = DL[t] + U^T DZ[t + 1]
    #   !!! DZ[t] = DH[t] * a'(Z[t])
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
        self.Wz = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Uz = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.bz = np.zeros(hidden_size)
        self.Wr = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Ur = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.br = np.zeros(hidden_size)
        self.Wh = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Uh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.bh = np.zeros(hidden_size)

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
        _check_size("mtn, mh, nh", X, Hinit, self.Wz)
        m, t, n = X.shape
        Xz = X @ self.Wz + self.bz
        Xr = X @ self.Wr + self.br
        Xh = X @ self.Wh + self.bh
        H = np.empty_like(Xz)
        for i in range(0, t):
            Sz = Xz[:, i, :] + Hinit @ self.Uz
            Z = sigmoid(Sz)
            Sr = Xr[:, i, :] + Hinit @ self.Ur
            R = sigmoid(Sr)
            Sh = Xh[:, i, :] + (R * Hinit) @ self.Uh
            Hnew = self.forward_activation(Sh)
            Hinit = (1 - Z) * Hinit + Z * Hnew
            H[:, i, :] = Hinit
        return H

    def backward(self, X, H, DL, Dinit, Hinit):
        """Backward step: return derivatives computed for all time steps.

        Parameters
        ----------
        X : ndarray, shape (m, t, n)
            sequence of input features (m sequences, t time steps, n components).
        H : ndarray, shape (m, t, h)
            sequence of states obtained in the forward step.
        DL : ndarray, shape (m, t, h)
            derivative of the losses at each time steps with respect to the corresponding states.
        Dinit : ndarray, shape (m, h)
            derivative of the total loss with respect to the last logits obtained in the forward
            step.
        Hinit : ndarray, shape (m, h)
            intial state (one vector for each input sequence).

        Returns
        -------
        DZ : ndarray, shape (m, t, h)
            derivative of the losses at each time steps with respect to the logits.
        DX : ndarray, shape (m, t, n)
            derivative of the losses at each time steps with respect to the inputs.
        """

        _check_size("mtn, mth, mth, mh, mh", X, H, DL, Dinit, Hinit)
        m, t, n = H.shape
        Hold = np.concatenate((Hinit[:, None, :], H[:, :-1, :]), 1)
        Sz = X @ self.Wz + Hold @ self.Uz + self.bz
        Z = sigmoid(Sz)
        Sr = X @ self.Wr + Hold @ self.Ur + self.br
        R = sigmoid(Sr)
        Sh = X @ self.Wh + (Hold * R) @ self.Uh + self.bh
        Hnew = self.forward_activation(Sh)
        DHnew = DL * Z
        DZ = DL * (Hnew - Hold)
        DSz = DZ * Z * (1 - Z)
        backact = self.backward_activation(Hnew)
        DSh = DL * Z * backact
        DR = ((DSh * Hold) @ self.Uh.T)
        breakpoint()
        DSr = DR * R * (1 - R)
        DH = np.empty_like(H)
        DH[:, -1, :] = Dinit * (1 - Z[:, -1, :])
        for i in range(t - 2, -1, -1):
            DH[:, i, :] = DH[:, i + 1, :] * (1 - Z[:, i, :])
            DH[:, i, :] += DSz[:, i + 1, :] @ self.Uz.T
            DH[:, i, :] += DSr[:, i + 1, :] @ self.Ur.T 
            DH[:, i, :] += (Hold[:, i, :] * Z[:, i, :] * backact[:, i, :]) @ self.Uh.T
        DX = DSz @ self.Wz.T + DSr @ self.Wr.T + DSh @ self.Wh.T
        return DZ, DX

    def forward_activation(self, X):
        """Activation function."""
        return np.tanh(X)

    def backward_activation(self, A):
        """Derivative of the activation, given the activation values."""
        return 1 - (A ** 2)

    def parameters(self):
        """List of parameters of the cell."""
        return (self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh)

    def parameters_grad(self, X, H, DZ, DZinit):
        """Derivative of the total loss with respect to the parameters of the cell."""
        n, h = self.W.shape
        DV = H[:, :-1, :].reshape(-1, h).T @ DZ[:, 1:, :].reshape(-1, h)
        DV += H[:, -1, :].T @ DZinit
        Db = DZ.sum((0, 1))
        DW = X.reshape(-1, n).T @ DZ.reshape(-1, h)
        return (DW, DV, Db)
