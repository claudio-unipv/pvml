import numpy as np
import pvml


def _check_gradient(cnn, eps=1e-6, imsz=25, bsz=2):
    """Numerical check gradient computations."""
    insz = cnn.weights[0].shape[2]
    outsz = cnn.weights[-1].shape[3]
    X = np.random.randn(bsz, imsz, imsz, insz)
    Y = np.random.randint(0, outsz, (bsz,))
    A = cnn.forward(X)
    D = cnn.backward(Y, A)
    L = cnn.loss(Y, A[-1])
    for XX, W, DD, s in zip(A, cnn.weights, D, cnn.strides):
        grad_W = pvml.cnn._convolution_derivative(XX, DD, W.shape[0],
                                                  W.shape[1], s, s)
        GW = np.empty_like(W)
        for idx in np.ndindex(*W.shape):
            bak = W[idx]
            W[idx] += eps
            A1 = cnn.forward(X)
            L1 = cnn.loss(Y, A1[-1])
            W[idx] = bak
            GW[idx] = (L1 - L) / eps
        err = np.abs(GW - grad_W).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4
    for b, DD in zip(cnn.biases, D):
        grad_b = DD.sum(2).sum(1).sum(0)
        Gb = np.empty_like(b)
        for idx in np.ndindex(*b.shape):
            bak = b[idx]
            b[idx] += eps
            A1 = cnn.forward(X)
            L1 = cnn.loss(Y, A1[-1])
            b[idx] = bak
            Gb[idx] = (L1 - L) / eps
        err = np.abs(Gb - grad_b).max()
        print(err, "OK" if err < 1e-4 else "")
        assert err < 1e-4


if __name__ == "__main__":
    cnn = pvml.CNN([4, 6], [2], [1])
    _check_gradient(cnn)
    # cnn = pvml.CNN([8, 7, 6], [1, 1], [1, 1])
    # _check_gradient(cnn)
    cnn = pvml.CNN([3, 8, 7, 5], [5, 4, 3], [2, 1, 1])
    _check_gradient(cnn)
