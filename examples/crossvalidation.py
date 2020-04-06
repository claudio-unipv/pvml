import numpy as np
import pvml
import sys


def ksvm_cross_validation(k, X, Y, kfun, kparam, lambda_, lr=1e-3, steps=10000):
    m = X.shape[0]
    # Randomly assign a fold index to each sample.
    folds = np.arange(m) % k
    np.random.shuffle(folds)
    correct_predictions = 0
    # For each fold...
    for fold in range(k):
        Xtrain = X[folds != fold, :]
        Ytrain = Y[folds != fold]
        # Train a model
        alpha, b = pvml.ksvm_train(Xtrain, Ytrain, kfun, kparam,
                                   lambda_, lr=lr, steps=steps)
        # Evaluate the model on the left-out fold
        Xval = X[folds == fold, :]
        Yval = Y[folds == fold]
        pred, _ = pvml.ksvm_inference(Xval, Xtrain, alpha, b, kfun, kparam)
        print((pred == Yval).mean())
        correct_predictions += (pred == Yval).sum()
    return correct_predictions / m


def _main():
    X, Y = pvml.load_dataset(sys.argv[1])
    accuracy = ksvm_cross_validation(5, X, Y, "rbf", 1, 1e-3)
    print("Accuracy: %f%%" % (accuracy * 100))


if __name__ == "__main__":
    _main()
