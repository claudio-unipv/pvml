#!/usr/bin/env python3


import pvml
import numpy as np
import matplotlib.pyplot as plt


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
    weights = weights[0]
    v = np.abs(weights).max()
    count = weights.shape[-1]
    rows = max([min(x, count // x) for x in range(1, count // 2)
                if count % x == 0])
    cols = count // rows
    plt.figure(2)
    plt.clf()
    for i in range(count):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(weights[:, :, 0, i], cmap=plt.cm.seismic, vmin=-v,
                   vmax=v)
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
            plt.text(j - 0.25, i, txt)
    plt.title("Confusion matrix")


def main():
    Xtrain, Ytrain = pvml.load_dataset("mnist_train")
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest, Ytest = pvml.load_dataset("mnist_test")
    Xtest = Xtest.reshape(-1, 28, 28, 1)

    # meanvar normalization (use the same statistics for all the pixels)
    mean = Xtrain.mean()
    std = Xtrain.std()
    Xtrain = (Xtrain - mean) / std
    Xtest = (Xtest - mean) / std
    
    plt.ion()
    batch_sz = 100
    epochs = 75

    network = pvml.CNN([1, 12, 32, 48, 10], [7, 3, 3, 3], [2, 2, 1, 1])
    A = network.forward(np.empty((1, 28, 28, 1)))
    print("Neurons:", " -> ".join(("x".join(map(str, a.shape[1:]))) for
                                  a in A))
    parameters = (sum(w.size for w in network.weights) +
                  sum(b.size for b in network.biases))
    print(parameters, "parameters")
    errors = [[], []]
    for epoch in range(1, epochs + 1):
        steps = Xtrain.shape[0] // batch_sz
        network.train(Xtrain, Ytrain, lr=1e-3, lambda_=1e-5,
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
    np.savetxt("errors-l1.txt", np.array(errors).T, fmt="%.2f")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
