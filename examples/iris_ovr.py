import numpy as np
import pvml


def ovr_train(X, Y, k):
    W = np.zeros((2, 3))
    b = np.zeros(3)
    for c in range(k):
        Y1 = (Y == c).astype(int)
        W[:, c], b[c] = pvml.svm_train(X, Y1, 0, lr=0.1, steps=100000)
    return W, b


def ovr_inference(X, W, b):
    Z = X @ W + b.T
    return Z.argmax(1)


X, Y = pvml.load_dataset("iris")
W, b = ovr_train(X, Y, 3)
predictions =ovr_inference(X, W, b)
accuracy = (predictions == Y).mean()
print(accuracy * 100)
