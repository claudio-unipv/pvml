#!/usr/bin/env python3

import pvml
import numpy as np
import matplotlib.pyplot as plt


def first_layer_image(weights):
    count = weights.shape[1]
    rows = max([min(x, count // x) for x in range(1, count // 2)
                if count % x == 0])
    cols = count // rows
    im = weights.reshape(28, 28, rows, cols).transpose(2, 0, 3, 1)
    im = im.reshape(28 * rows, -1)
    return im


def plot_errors(errors):
    plt.figure(1)
    plt.clf()
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.legend(["Training", "Test"])
    plt.xlabel("Epochs")
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


def show_confusion_matrix(Y, predictions):
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3)
    plt.clf()
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Blues)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")


def main():
    Xtrain, Ytrain = pvml.load_dataset("mnist_train")
    Xtest, Ytest = pvml.load_dataset("mnist_test")

    plt.ion()
    batch_sz = 100
    epochs = 250

    network = pvml.MLP([Xtrain.shape[1], 128, 64, 10])
    errors = [[], []]
    for epoch in range(1, epochs + 1):
        steps = Xtrain.shape[0] // batch_sz
        network.train(Xtrain, Ytrain, lr=1e-4, lambda_=1e-5,
                      steps=steps, batch=batch_sz)
        predictions = network.inference(Xtrain)[0]
        training_error = (predictions != Ytrain).mean()
        predictions = network.inference(Xtest)[0]
        test_error = (predictions != Ytest).mean()
        errors[0].append(100 * training_error)
        errors[1].append(100 * test_error)
        msg = "Epoch {}, Training err. {:.2f}, Test err. {:.2f}"
        print(msg.format(epoch, 100 * training_error, 100 * test_error))
        plot_errors(errors)
        show_weights(network.weights)
        show_confusion_matrix(Ytest, predictions)
        plt.pause(0.05)
    network.save("mnist_network.npz")
    np.savetxt("mnist_errors.txt", np.array(errors).T, fmt="%.2f")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
