import numpy as np
from .checks import _check_size, _check_labels
from .mlp import relu
from .multinomial_logistic import softmax, cross_entropy
from .logistic_regression import sigmoid


class RNN:
    """Recurrent Neural Network."""
    def __init__(self, neuron_counts, cell_type="basic"):
        """Create the RNN."""
        self.neuron_counts = neuron_counts
        connections = zip(neuron_counts[:-2], neuron_counts[1:-1])
        cell_factory = _CELL_TYPES[cell_type.lower()]
        self.cells = [cell_factory(n1, n2) for n1, n2 in connections]
        self.W = (np.random.randn(neuron_counts[-2], neuron_counts[-1]) *
                  np.sqrt(2 / neuron_counts[-2]))
        self.b = np.zeros(neuron_counts[-1])
        self.reset_momentum()

    def reset_momentum(self):
        """Set to zero the accumulators for momentum."""
        self.up_W = np.zeros_like(self.W)
        self.up_b = np.zeros_like(self.b)
        self.up_p = [[np.zeros_like(p) for p in c.parameters()] for c in self.cells]

    def forward(self, X, inits=None):
        """Forward step of the RNN.

        Parameters
        ----------
        X : ndarray, shape (m, t, n)
            sequence of input features (m sequences, t time steps, n components).
        inits : list
            list of initial states, one for each RNN layer (default use zeros).

        Returns
        -------
        P : ndarray, shape (m, t, k)
            sequence of output estimates (m sequences, t time steps, k output estimates).
        """
        m = X.shape[0]
        if inits is None:
            inits = [np.zeros((m, h)) for h in self.neuron_counts[1:-1]]
        H = X
        for cell, init in zip(self.cells, inits):
            H = cell.forward(H, init)
        V = H @ self.W + self.b
        P = self.activation(V)
        self.H = H
        self.P = P
        return P

    def backward(self, Y):
        """Backward step of the RNN.

        Parameters
        ----------
        Y : ndarray, shape (m,) or (m, t)
            target output (integer class labels, one per sequence or one per sequence element).

        Returns
        -------
        D : ndarray, shape (m, t, n)
            derivative of the losses at each time steps with respect to the inputs.
        """
        DV = self.activation_backward(self.P, Y)
        D = DV @ self.W.T
        for cell in self.cells[::-1]:
            D = cell.backward(D)
        self.DV = DV
        return D

    def train(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99,
              steps=10000, batch=None):
        """Train the network.

        Apply multiple steps of backpropagation to train the network.

        Parameters
        ----------
        X : ndarray, shape (m, t, n)
            input sequences (m sequences of length t and with n features).
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
        """Backpropagation algorithm

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
        """
        self.forward(X)
        self.backward(Y)
        h, k = self.W.shape
        grad_W = self.H.reshape(-1, h).T @ self.DV.reshape(-1, k) + lambda_ * self.W
        grad_b = self.DV.sum((0, 1))
        grads = [cell.parameters_grad() for cell in self.cells]

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
        """Final activation function."""
        m, t, n = V.shape
        return softmax(V.reshape(-1, n)).reshape(m, t, n)

    def activation_backward(self, P, Y):
        """Derivatives of the final activation function."""
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


class RNNAbstractCell:
    """A cell for building RNNs."""

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
        raise NotImplementedError

    def backward(self, DL):
        """Backward step: return derivatives computed for all time steps.

        Parameters
        ----------
        DL : ndarray, shape (m, t, h)
            derivative of the losses at each time steps with respect to the corresponding states.

        Returns
        -------
        DX : ndarray, shape (m, t, n)
            derivative of the losses at each time steps with respect to the inputs.
        """
        raise NotImplementedError

    def parameters(self):
        """List of parameters of the cell.

        Returns
        -------
        params : list
            list of arrays representing the parameters of the cell.
        """
        raise NotImplementedError

    def parameters_grad(self):
        """Derivative of the total loss with respect to the parameters of the cell.

        Returns
        -------
        params : list
            list of arrays representing the derivatives of the parameters of the cell
            in an order consistent with that of the 'parameters' method.
        """
        raise NotImplementedError


class RNNBasicCell(RNNAbstractCell):
    """A Basic RNN cell."""
    def __init__(self, input_size, hidden_size):
        """Initialize the cell.

        Parameters
        ----------
        input_size : int
            number of input neurons.
        hidden_size : int
            number of hidden neurons.
        """
        # Parameters
        self.W = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.U = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b = np.zeros(hidden_size)
        # State
        self.X = np.empty((0, 0, input_size))
        self.Z = np.empty((0, 0, hidden_size))
        self.H = np.empty((0, 0, hidden_size))

    def forward(self, X, Hinit):
        # Given X:
        #   Z[t] = W X[t] + U H[t - 1] + b
        #   H[t] = a(Z[t])
        _check_size("mtn, mh, nh", X, Hinit, self.W)
        m, t, n = X.shape
        X1 = X @ self.W + self.b
        H = np.empty_like(X1)
        Z = np.empty_like(X1)
        for i in range(0, t):
            Z[:, i, :] = X1[:, i, :] + Hinit @ self.U
            H[:, i, :] = self.forward_activation(Z[:, i, :])
            Hinit = H[:, i, :]
        self.X = X
        self.H = H
        self.Z = Z
        return H

    def backward(self, DL):
        # Given DL[t] = dLoss[t] / dH[t]:
        #   DH[t] = DL[t] + U^T DZ[t + 1]
        #   DZ[t] = DH[t] * a'(Z[t])
        _check_size("mth, mth", DL, self.H)
        t = DL.shape[1]
        DH = DL.copy()
        DZ = np.empty_like(DL)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                DH[:, i, :] += DZ[:, i + 1, :] @ self.U.T
            DZ[:, i, :] = DH[:, i, :] * self.backward_activation(self.H[:, i, :])
        DX = DZ @ self.W.T
        self.DH = DH
        self.DZ = DZ
        self.DX = DX
        return DX

    def forward_activation(self, X):
        """Activation function."""
        return relu(X)

    def backward_activation(self, A):
        """Derivative of the activation, given the activation values."""
        return (A > 0).astype(float)

    def parameters(self):
        return (self.W, self.U, self.b)

    def parameters_grad(self):
        # dTotalLoss / dW = sum( DZ[t] X[t]^T )
        # dTotalLoss / dU = sum( DZ[t] H[t - 1]^T )
        # dTotalLoss / db = sum( DZ[t] )
        n, h = self.W.shape
        DW = self.X.reshape(-1, n).T @ self.DZ.reshape(-1, h)
        DU = self.H[:, :-1, :].reshape(-1, h).T @ self.DZ[:, 1:, :].reshape(-1, h)
        Db = self.DZ.sum(1).sum(0)
        return (DW, DU, Db)


class GRUCell(RNNAbstractCell):
    """Gated Recurring Unit cell."""

    def __init__(self, input_size, hidden_size):
        """Initialize the cell.

        Parameters
        ----------
        input_size : int
            number of input neurons.
        hidden_size : int
            number of hidden neurons.
        """
        # Parameters
        self.Wz = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Uz = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.bz = np.zeros(hidden_size)
        self.Wr = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Ur = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.br = np.zeros(hidden_size)
        self.Wh = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.Uh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.bh = np.zeros(hidden_size)
        # Cell state variables
        self.X = np.empty((0, 0, input_size))
        self.R = np.empty((0, 0, hidden_size))
        self.Sr = np.empty((0, 0, hidden_size))
        self.Z = np.empty((0, 0, hidden_size))
        self.Sz = np.empty((0, 0, hidden_size))
        self.Hnew = np.empty((0, 0, hidden_size))
        self.Sh = np.empty((0, 0, hidden_size))
        self.H = np.empty((0, 0, hidden_size))

    def forward(self, X, Hinit):
        #   Sz[t] = Wz X[t] + Uz H[t - 1] + bz
        #   Z[t] = sigmoid(Sz[t])
        #   Sr[t] = Wr X[t] + Ur H[t - 1] + br
        #   R[t] = sigmoid(Sr[t])
        #   Sh[t] = Wh X[t] + Uh (R[t] * H[t - 1]) + bh
        #   H'[t] = a(Sh[t])
        #   H[t] = (1 - Z[t]) * H[t - 1] + Z[t] * H'[t]
        _check_size("mtn, mh, nh", X, Hinit, self.Wz)
        m, t, n = X.shape
        Xz = X @ self.Wz + self.bz
        Xr = X @ self.Wr + self.br
        Xh = X @ self.Wh + self.bh
        H = np.empty_like(Xz)
        R = np.empty_like(H)
        Sr = np.empty_like(H)
        Z = np.empty_like(H)
        Sz = np.empty_like(H)
        Hnew = np.empty_like(H)
        Sh = np.empty_like(H)
        self.Hinit = Hinit.copy()
        for i in range(0, t):
            Sz[:, i, :] = Xz[:, i, :] + Hinit @ self.Uz
            Z[:, i, :] = sigmoid(Sz[:, i, :])
            Sr[:, i, :] = Xr[:, i, :] + Hinit @ self.Ur
            R[:, i, :] = sigmoid(Sr[:, i, :])
            Sh[:, i, :] = Xh[:, i, :] + (R[:, i, :] * Hinit) @ self.Uh
            Hnew[:, i, :] = self.forward_activation(Sh[:, i, :])
            Hinit = (1 - Z[:, i, :]) * Hinit + Z[:, i, :] * Hnew[:, i, :]
            H[:, i, :] = Hinit
        self.X = X.copy()
        self.R = R
        self.Sr = Sr
        self.Z = Z
        self.Sz = Sz
        self.Hnew = Hnew
        self.Sh = Sh
        self.H = H
        return H

    def backward(self, DL):
        # Given DL[t] = dLoss[t] / dH[t]:
        #   DH[t] = DL[t] + Uz^T DSz[t + 1] + Ur^T DSr[t + 1]
        #         + (Uh^T DSz[t + 1]) * R[t + 1] + DH[t + 1] * (1 - Z[t + 1])
        #
        #   DH'[t] = DH[t] * Z[t]
        #   DSh[t] = DH'[t] * a'(H[t])
        #   DZ[t] = DH[t] * (H'[t] - H[t - 1])
        #   DSz[t] = DZ[t] * Z[t] * (1 - Z[t])
        #   DR[t] = (Uh^T DSh[t]) H[t - 1]
        #   DSr[t] = DR[t] * R[t] * (1 - R[t])
        #
        #   DX[t] = Wz^T DSz + Wr^T DSr + Wh^T DSh
        _check_size("mth, mth", self.H, DL)
        m, t, h = self.H.shape
        DH = DL.copy()
        DX = np.empty_like(self.X)
        DHnew = np.empty_like(self.H)
        DSh = np.empty_like(self.H)
        DZ = np.empty_like(self.H)
        DSz = np.empty_like(self.H)
        DR = np.empty_like(self.H)
        DSr = np.empty_like(self.H)
        for i in range(t - 1, -1, -1):
            # Compute DH
            if i < t - 1:
                DH[:, i, :] += DSz[:, i + 1, :] @ self.Uz.T
                DH[:, i, :] += DSr[:, i + 1, :] @ self.Ur.T
                DH[:, i, :] += (DSh[:, i + 1, :]  @ self.Uh.T) * self.R[:, i + 1, :]
                DH[:, i, :] += (1 - self.Z[:, i + 1, :]) * DH[:, i + 1, :]
            # Update the deratives of gates and logits
            DHnew[:, i, :] = DH[:, i, :] * self.Z[:, i, :]
            DSh[:, i, :] = DHnew[:, i, :] * self.backward_activation(self.Hnew[:, i, :])
            Hold = (self.Hinit if i == 0 else self.H[:, i - 1, :])
            DZ[:, i, :] = DH[:, i, :] * (self.Hnew[:, i, :] - Hold)
            DSz[:, i, :] = DZ[:, i, :] * self.Z[:, i, :] * (1 - self.Z[:, i, :])
            DR[:, i, :] = (DSh[:, i, :]  @ self.Uh.T) * Hold
            DSr[:, i, :] = DR[:, i, :] * self.R[:, i, :] * (1 - self.R[:, i, :])
        # Compute DX
        DX = DSz @ self.Wz.T + DSr @ self.Wr.T + DSh @ self.Wh.T
        self.DH = DH
        self.DX = DX
        self.DHnew = DHnew
        self.DSh = DSh
        self.DZ = DZ
        self.DSz = DSz
        self.DR = DR
        self.DSr = DSr
        return DX

    def forward_activation(self, X):
        """Activation function."""
        return np.tanh(X)

    def backward_activation(self, A):
        """Derivative of the activation, given the activation values."""
        return 1 - (A ** 2)

    def parameters(self):
        """List of parameters of the cell."""
        return (self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh)

    def parameters_grad(self):
        """Derivative of the total loss with respect to the parameters of the cell."""
        # dTotalLoss / dWz = sum( DSz[t] X[t]^T )
        # dTotalLoss / dUz = sum( DSz[t] H[t - 1]^T )
        # dTotalLoss / dbz = sum( DSz[t] )
        # dTotalLoss / dWr = sum( DSr[t] X[t]^T )
        # dTotalLoss / dUr = sum( DSr[t] H[t - 1]^T )
        # dTotalLoss / dbr = sum( DSr[t] )
        # dTotalLoss / dWh = sum( DSh[t] X[t]^T )
        # dTotalLoss / dUh = sum( DSh[t] (R[t] * H[t - 1])^T )
        # dTotalLoss / dbh = sum( DSh[t] )
        n, h = self.Wz.shape
        Hold = np.concatenate((self.Hinit[:, None, :], self.H[:, :-1, :]), 1)
        Hold = Hold.reshape(-1, h)
        DWz = self.X.reshape(-1, n).T @ self.DSz.reshape(-1, h)
        DWr = self.X.reshape(-1, n).T @ self.DSr.reshape(-1, h)
        DWh = self.X.reshape(-1, n).T @ self.DSh.reshape(-1, h)
        DUz = Hold.T @ self.DSz.reshape(-1, h)
        DUr = Hold.T @ self.DSr.reshape(-1, h)
        DUh = (self.R.reshape(-1, h) * Hold).T @ self.DSh.reshape(-1, h)
        Dbz = self.DSz.sum(1).sum(0)
        Dbr = self.DSr.sum(1).sum(0)
        Dbh = self.DSh.sum(1).sum(0)
        return (DWz, DUz, Dbz, DWr, DUr, Dbr, DWh, DUh, Dbh)


# Factory used to select the cell type from the RNN constructor
_CELL_TYPES = {"basic": RNNBasicCell, "gru": GRUCell}
