#!/usr/bin/env python3

import dataset
import numpy as np
import mlp
import matplotlib.pyplot as plt


def first_layer_image(weights):
    count = weights.shape[1]
    cols = max([min(x, count // x) for x in range(1, count // 2)
                if count % x == 0])
    rows = count // cols
    im = weights.reshape(28, 28, rows, cols).transpose(2, 0, 3, 1)
    im = im.reshape(28 * rows, -1)
    return im


def plot_errors(errors):
    plt.figure(1)
    plt.clf()
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.legend(["Training", "Test"])
    plt.xlabel("Epocs")
    plt.ylabel("Error (%)")
    plt.title("Classification error")


def show_weights(weights):
    im = first_layer_image(weights[0])
    plt.figure(2)
    plt.clf()
    plt.subplot(1, len(weights), 1)
    v = np.abs(im).max()
    plt.imshow(im, cmap=plt.cm.seismic, vmin=-v, vmax=v)
    for i in range(1, len(weights)):
        plt.subplot(1, len(weights), i + 1)
        v = np.abs(weights[i]).max()
        plt.imshow(weights[i], cmap=plt.cm.seismic, vmin=-v, vmax=v)
    plt.title("Weights")


def show_errors(X, Y, predictions, k=100):
    errors = (Y != predictions).nonzero()[0]
    np.random.shuffle(errors)
    errors = errors[:k]
    X = X[errors]
    count = X.shape[0]
    cols = max([min(x, count // x) for x in range(1, 1 + count)
                if count % x == 0])
    rows = count // cols
    im = X.reshape(rows, cols, 28, 28).transpose(0, 2, 1, 3)
    im = im.reshape(rows * 28, -1)
    plt.figure(3)
    plt.clf()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.title("Errors")


def main():
    Xtrain, Ytrain = dataset.load_dataset("mnist_train")
    Xtest, Ytest = dataset.load_dataset("mnist_test")

    plt.ion()
    batch_sz = 32
    epocs = 200

    network = mlp.MLP([Xtrain.shape[1], 64, 64, 10])
    errors = [[], []]
    for epoc in range(1, epocs + 1):
        steps = Xtrain.shape[0] // batch_sz
        network.train(Xtrain, Ytrain, lr=1e-4, lambda_=1e-5,
                      steps=steps, batch=batch_sz)
        predictions = network.inference(Xtrain)[0]
        training_error = (predictions != Ytrain).mean()
        predictions = network.inference(Xtest)[0]
        test_error = (predictions != Ytest).mean()
        errors[0].append(100 * training_error)
        errors[1].append(100 * test_error)
        print(epoc, 100 * training_error, 100 * test_error)
        plot_errors(errors)
        show_weights(network.weights)
        show_errors(Xtest, Ytest, predictions)
        plt.pause(0.05)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
