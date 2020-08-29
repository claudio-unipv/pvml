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
    plt.ylabel("Error (MSE)")
    plt.title("Reconstruction error")


def show_reconstruction(before, after):
    k = 10
    k2 = k ** 2
    m = before.shape[0]
    before = before[::(m // k2)][:k2]
    after = after[::(m // k2)][:k2]
    plt.figure(2)
    plt.clf()
    plt.subplot(1, 3, 1)
    before = before.reshape(k, k, 28, 28).transpose(0, 2, 1, 3).reshape(k * 28, k * 28)
    plt.imshow(before, cmap="gray")
    plt.title("Input")
    plt.subplot(1, 3, 2)
    after = after.reshape(k, k, 28, 28).transpose(0, 2, 1, 3).reshape(k * 28, k * 28)
    plt.imshow(after, cmap="gray")
    plt.title("Output")
    plt.subplot(1, 3, 3)
    error = np.abs(before - after)
    plt.imshow(error, cmap="hot", vmin=0, vmax=1)
    plt.title("Error")


class AutoEncoder(pvml.MLP):
    def forward_output_activation(self, X):
        """Activation function of the output layer."""
        return pvml.sigmoid(X)

    def backward_output_activation(self, Y, P):
        """Derivative of the activation function of output layer."""
        # Derivative of the composition of the MSE (-2 * (Y - P))
        # combined with the sigmoid (P * (1 - P)).
        return -2 * (Y - P) * P * (1 - P) / Y.shape[0]

    def loss(self, Y, P):
        """Compute the average MSE."""
        return ((Y - P) ** 2).mean()


def main():
    Xtrain, Ytrain = pvml.load_dataset("mnist_train")
    Xtest, Ytest = pvml.load_dataset("mnist_test")

    plt.ion()
    batch_sz = 100
    epochs = 250

    network = AutoEncoder([Xtrain.shape[1], 100, 10, 100, Xtrain.shape[1]])
    errors = [[], []]
    for epoch in range(1, epochs + 1):
        steps = Xtrain.shape[0] // batch_sz
        network.train(Xtrain, Xtrain, lr=1e-3, lambda_=1e-5,
                      steps=steps, batch=batch_sz)
        reconstruction = network.forward(Xtrain)[-1]
        training_error = network.loss(Xtrain, reconstruction)
        reconstruction = network.forward(Xtest)[-1]
        test_error = network.loss(Xtest, reconstruction)
        errors[0].append(training_error)
        errors[1].append(test_error)
        msg = "Epoch {}, Training err. {:.3f}, Test err. {:.3f}"
        print(msg.format(epoch, training_error, test_error))
        plot_errors(errors)
        show_reconstruction(Xtest, reconstruction)
        plt.pause(0.05)
    network.save("mnist_autoencoder_network.npz")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
