#!/usr/bin/env python3

import pvml
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Classification demo")
    a = parser.add_argument
    a("-r", "--lr", type=float, default=0.001,
      help="learning rate (%(default)g)")
    a("-l", "--lambda", type=float, dest="lambda_", default=0,
      help="regularization coefficient (%(default)g)")
    a("-s", "--steps", type=int, default=10000,
      help="maximum number of training iterations (%(default)d)")
    a("-p", "--plot-every", type=int, default=100,
      help="frequency of plotting training data (%(default)d)")
    a("-t", "--test", help="test set")
    a("--seed", type=int, default=171956, help="Random seed")
    a("--dump", action="store_true",
      help="Save the decision boundary and other data")
    a("--nodraw", action="store_true", help="Skip drawing the plots")
    a("--model", choices=_MODELS.keys(), default="logreg",
      help="Classification model")
    a("train", help="training set")
    return parser.parse_args()


class DemoModel:
    def __init__(self, args, binary):
        self.lr = args.lr
        self.lambda_ = args.lambda_
        self.binary = binary
        self.plot_every = args.plot_every
    
    def train(self, X, Y, Xtest, Ytest, steps):
        st = self.plot_every
        iterations = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        print("Step Train", "" if Xtest is None else "Test")
        for step in range(st, steps + st, st):
            self.train_step(X, Y, st)
            iterations.append(step)
            P = self.inference(X)
            train_acc.append(100 * (np.rint(P) == Y).mean())
            train_loss.append(self.loss(Y, P))
            plt.figure(0)
            plt.clf()
            plt.title("Accuracy (%)")
            plt.xlabel("Iterations")
            plt.plot(iterations, train_acc)
            if train_loss[-1] is not None:
                plt.figure(1)
                plt.clf()
                plt.title("Loss")
                plt.xlabel("Iterations")
                plt.plot(iterations, train_loss)
            if Xtest is not None:                
                Ptest = self.inference(Xtest)
                test_acc.append(100 * (np.rint(Ptest) == Ytest).mean())
                test_loss.append(self.loss(Ytest, Ptest))
                plt.figure(0)
                plt.plot(iterations, test_acc)
                plt.legend(["train", "test"])
                if test_loss[-1] is not None:
                    plt.figure(1)
                    plt.plot(iterations, test_loss)
                    plt.legend(["train", "test"])
            if X.shape[1] == 2:
                plt.figure(2)
                plt.clf()
                plt.title("Training set")
                self.plot_data(X, Y)
                if Xtest is not None:
                    plt.figure(3)
                    plt.clf()
                    plt.title("Test set")
                    self.plot_data(Xtest, Ytest)
            if Xtest is None:
                print("{} {:.2f}%".format(step, train_acc[-1]))
            else:
                print("{} {:.2f}% {:.2f}%".format(step, train_acc[-1],
                                                  test_acc[-1]))
            plt.pause(0.0001)

    def plot_data(self, X, Y, resolution=20):
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        ax = np.linspace(xmin, xmax, resolution)
        ay = np.linspace(ymin, ymax, resolution)
        gx, gy = np.meshgrid(ax, ay)
        data = np.vstack((gx.reshape(-1), gy.reshape(-1))).T
        v = self.inference(data).reshape(gx.shape)
        values = (np.sort(np.unique(Y)) + 0.5)[:-1]
        data = np.stack((gx, gy, v), -1)
        plt.contour(gx, gy, v, values, cmap=plt.cm.coolwarm)
        
    def train_step(self, X, Y, steps):
        pass

    def inference(self, X):
        pass

    def loss(self, Y, P):
        pass
    

_MODELS = {}


def _register_model(name):
    def f(cls):
        _MODELS[name] = cls
        return cls
    return f


@_register_model("logreg")    
class LogisticRegressionModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, True)
        self.w = None
        self.b = 0
        
    def train_step(self, X, Y, steps):
        self.w, self.b = pvml.logreg_train(X, Y, lr=self.lr,
                                           steps=steps, init_w=self.w,
                                           init_b=self.b)

    def inference(self, X):
        return pvml.logreg_inference(X, self.w, self.b)

    def loss(self, Y, P):
        return pvml.cross_entropy(Y, P)
    

def main():
    args = parse_args()
    np.random.seed(args.seed)
    X, Y = pvml.load_dataset(args.train)
    if args.test:
        Xtest, Ytest = pvml.load_dataset(args.test)
    else:
        Xtest, Ytest = None, None
    model = _MODELS[args.model](args)
    if model.binary:
        Y = (Y > 0).astype(np.int_)
        if Ytest is not None:
            Ytest = (Ytest > 0).astype(np.int_)
    plt.ion()
    model.train(X, Y, Xtest, Ytest, args.steps)
    plt.ioff()
    print("TRAINING COMPLETED")
    plt.show()

    
if __name__ == "__main__":
    main()
