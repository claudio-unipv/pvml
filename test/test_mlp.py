import numpy as np
import pvml


def check_gradient(mlp, eps=1e-6, bsz=2):
    """Numerically check gradient computations."""
    insz = mlp.weights[0].shape[0]
    outsz = mlp.weights[-1].shape[1]
    X = np.random.randn(bsz, insz)
    Y = np.random.randint(0, outsz, (bsz,))
    A = mlp.forward(X)
    D = mlp.backward(Y, A)
    L = mlp.loss(Y, A[-1])
    for XX, W, DD in zip(A, mlp.weights, D):
        grad_W = (XX.T @ DD)
        GW = np.empty_like(W)
        for idx in np.ndindex(*W.shape):
            bak = W[idx]
            W[idx] += eps
            A1 = mlp.forward(X)
            L1 = mlp.loss(Y, A1[-1])
            W[idx] = bak
            GW[idx] = (L1 - L) / eps
        err = np.abs(GW - grad_W).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4
    for b, DD in zip(mlp.biases, D):
        grad_b = DD.sum(0)
        Gb = np.empty_like(b)
        for idx in np.ndindex(*b.shape):
            bak = b[idx]
            b[idx] += eps
            A1 = mlp.forward(X)
            L1 = mlp.loss(Y, A1[-1])
            b[idx] = bak
            Gb[idx] = (L1 - L) / eps
        err = np.abs(Gb - grad_b).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4


if __name__ == "__main__":
    mlp = pvml.MLP([3, 10])
    check_gradient(mlp)
    mlp = pvml.MLP([4, 7, 4])
    check_gradient(mlp)
    mlp = pvml.MLP([5, 6, 8, 7])
    check_gradient(mlp)
