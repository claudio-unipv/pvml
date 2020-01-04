import numpy as np


def _im2col(X, kh, kw, sh, sw):
    m, h, w, n = X.shape
    shape = (m, (h - kh + 1) // sh, (w - kw + 1) // sw, kh, kw, n)
    tm, th, tw, tn = X.strides
    strides = (tm, sh * th, sw * tw, th, tw, tn)
    Y = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides,
                                        writeable=False)
    return Y.reshape(m, shape[1], shape[2], kh * kw * n)


def convolution2d(X, W, sh, sw):
    # X: (m, h, w, n)
    # W: (sh, sw, n, c)
    Y = _im2col(X, W.shape[0], W.shape[1], sh, sw)
    Z = Y.reshape(-1, Y.shape[3]) @ W.reshape(-1, W.shape[3])
    print(Z.shape)
    return Z.reshape(Y.shape[0], Y.shape[1], Y.shape[2], -1)


X = np.arange(49 * 4).reshape(1, 7, 7, 4)
print(X[0, :, :, 0])
W = np.zeros((3, 3, 4, 2))
W[1, 0, 0, 0] = -1
W[1, 2, 0, 0] = 1
W[0, 0, 0, 1] = -1


Z = convolution2d(X, W, 2, 1)
print(Z[0, :, : ,0])
print(Z[0, :, : ,1])
