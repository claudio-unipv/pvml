#!/usr/bin/env python3

import pvml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import zip_longest


_NORMALIZATION = {
    "none": lambda *X: (X[0] if len(X) == 1 else X),
    "meanvar": pvml.meanvar_normalization,
    "minmax": pvml.minmax_normalization,
    "maxabs": pvml.maxabs_normalization,
    "l2": pvml.l2_normalization,
    "l1": pvml.l1_normalization,
    "whitening": pvml.whitening,
    "pca": pvml.pca
}


def parse_args():
    parser = argparse.ArgumentParser("Classification demo")
    a = parser.add_argument
    a("-r", "--lr", type=float, default=0.01,
      help="learning rate (%(default)g)")
    a("-l", "--lambda", type=float, dest="lambda_", default=0,
      help="regularization coefficient (%(default)g)")
    a("-s", "--steps", type=int, default=10000,
      help="maximum number of training iterations (%(default)d)")
    a("-p", "--plot-every", type=int, default=100,
      help="frequency of plotting training data (%(default)d)")
    a("-t", "--test", help="test set")
    a("-f", "--features", help="Comma-separated feature columns")
    a("-n", "--normalization", choices=_NORMALIZATION.keys(),
      default="none", help="Feature normalization")
    a("-c", "--class", type=int, default=-1, dest="class_",
      help="Class column")
    a("--seed", type=int, default=171956,
      help="Random seed")
    a("--confusion-matrix", "-C", action="store_true",
      help="Show the confusion matrix.")
    a("--dump", action="store_true",
      help="Save the decision boundary and other data")
    a("--nodraw", action="store_true",
      help="Skip drawing the plots")
    a("-m", "--model", choices=_MODELS.keys(), default="logreg",
      help="Classification model")
    a("-k", "--kernel", choices=["rbf", "polynomial"], default="rbf",
      help="Kernel function")
    a("--kernel-param", type=float, default=2,
      help="Parameter of the kernel")
    a("--knn-k", type=int, default=0, help="KNN neighbors (default auto)")
    a("--classtree-minsize", type=int, default=1,
      help="Classification tree minimum node size (%(default)d)")
    a("--classtree-diversity", default="gini",
      choices=["gini", "entropy", "error"],
      help="Classification tree diversity function (%(default)s)")
    a("--classtree-cv", type=int, default=5,
      help="Cross-validation folds used for pruning (%(default)d)")
    a("--mlp-hidden", default="",
      help="Comma-separated list of number of hidden neurons")
    a("--mlp-momentum", type=float, default=0.99,
      help="Momentum term (%(default)g)")
    a("--mlp-batch", type=int,
      help="Batch size (default: use all training data)")
    a("train", help="training set")
    return parser.parse_args()


class DemoModel:
    def __init__(self, args, binary, iterative=True):
        self.lr = args.lr
        self.lambda_ = args.lambda_
        self.binary = binary
        self.iterative = iterative
        self.plot_every = args.plot_every
        self.draw = not args.nodraw
        self.confusion_matrix = args.confusion_matrix
        self.dump = args.dump

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
            Z, P = self.inference(X)
            train_acc.append(100 * (Z == Y).mean())
            train_loss.append(self.loss(Y, P))
            if Xtest is not None:
                Ztest, Ptest = self.inference(Xtest)
                test_acc.append(100 * (Ztest == Ytest).mean())
                test_loss.append(self.loss(Ytest, Ptest))
            self.plot_curves(0, "Accuracy (%)", iterations, train_acc,
                             test_acc)
            self.plot_curves(1, "Loss", iterations, train_loss, test_loss)
            self.plot_confusion(4, "Confusion matrix (train)", Z, Y)
            if X.shape[1] == 2:
                self.plot_data(2, "Training set", X, Y)
                if Xtest is not None:
                    self.plot_data(3, "Test set", Xtest, Ytest)
            if Xtest is None:
                print("{} {:.2f}%".format(step, train_acc[-1]))
            else:
                self.plot_confusion(5, "Confusion matrix (test)", Ztest, Ytest)
                print("{} {:.2f}% {:.2f}%".format(step, train_acc[-1],
                                                  test_acc[-1]))
            plt.pause(0.0001)
            if not self.iterative or (self.draw and not plt.fignum_exists(0)):
                break
        if self.dump:
            with open("dump.txt", "wt") as f:
                for t in zip_longest(iterations, train_acc, test_acc,
                                     train_loss, test_loss):
                    row = (x if x is not None else "" for x in t)
                    print("{} {} {} {} {}".format(*row), file=f)

    def plot_curves(self, fignum, title, iters, train, test):
        train = [x for x in train if x is not None]
        test = [x for x in test if x is not None]
        if not self.draw or (not train and not test):
            return
        plt.figure(fignum)
        plt.clf()
        plt.title(title)
        plt.xlabel("Iterations")
        if train:
            plt.plot(iters, train)
        if test:
            plt.plot(iters, test)
            plt.legend(["train", "test"])

    def plot_data(self, fignum, title, X, Y, resolution=200):
        if not self.draw:
            return
        plt.figure(fignum)
        plt.clf()
        plt.title(title)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        ax = np.linspace(xmin, xmax, resolution)
        ay = np.linspace(ymin, ymax, resolution)
        gx, gy = np.meshgrid(ax, ay)
        data = np.vstack((gx.reshape(-1), gy.reshape(-1))).T
        v = self.inference(data)[1]
        if v.ndim == 1:
            v = v.reshape(gx.shape)
            plt.contour(gx, gy, v, [0.5], cmap=plt.cm.coolwarm)
        elif v.shape[1] == 2:
            v = v[:, 0] - v[:, 1]
            v = v.reshape(gx.shape)
            plt.contour(gx, gy, v, [0.0], cmap=plt.cm.coolwarm)
        else:
            values = np.arange(v.shape[1] - 1) + 0.5
            v = v.argmax(1)
            v = v.reshape(gx.shape)
            plt.contour(gx, gy, v, values, cmap=plt.cm.coolwarm)

    def plot_confusion(self, fignum, title, predictions, labels):
        if not self.draw or not self.confusion_matrix:
            return
        klasses = max(predictions.max(), labels.max()) + 1
        plt.figure(fignum)
        plt.clf()
        plt.title(title)
        cmat = np.bincount(klasses * labels + predictions,
                           minlength=klasses ** 2)
        cmat = cmat.reshape(klasses, klasses)
        cmat = 100 * cmat / np.maximum(1, cmat.sum(1, keepdims=True))
        im = plt.imshow(cmat, vmin=0, vmax=100, cmap="OrRd")
        plt.gca().set_xticks(np.arange(klasses))
        plt.gca().set_yticks(np.arange(klasses))
        colors = ("black", "white")
        for i in range(klasses):
            for j in range(klasses):
                val = cmat[i, j]
                color = (colors[0] if val < 50 else colors[1])
                im.axes.text(j, i, "%.1f" % val, color=color,
                             horizontalalignment="center",
                             verticalalignment="center")

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
        ret = pvml.logreg_l2_train(X, Y, lr=self.lr,
                                   lambda_=self.lambda_, steps=steps,
                                   init_w=self.w, init_b=self.b)
        self.w, self.b = ret

    def inference(self, X):
        P = pvml.logreg_inference(X, self.w, self.b)
        return (P > 0.5).astype(int), P

    def loss(self, Y, P):
        return pvml.binary_cross_entropy(Y, P)


@_register_model("logreg_l1")
class LogisticRegressionL1Model(LogisticRegressionModel):
    def train_step(self, X, Y, steps):
        ret = pvml.logreg_l1_train(X, Y, lr=self.lr,
                                   lambda_=self.lambda_, steps=steps,
                                   init_w=self.w, init_b=self.b)
        self.w, self.b = ret


@_register_model("ksvm")
class KernelSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, True)
        self.alpha = None
        self.b = 0
        self.Xtrain = None
        self.kfun = args.kernel
        self.kparam = args.kernel_param

    def train_step(self, X, Y, steps):
        self.Xtrain = X
        ret = pvml.ksvm_train(X, Y, self.kfun, self.kparam,
                              lr=self.lr, lambda_=self.lambda_,
                              steps=steps, init_alpha=self.alpha,
                              init_b=self.b)
        self.alpha, self.b = ret

    def inference(self, X):
        ret = pvml.ksvm_inference(X, self.Xtrain, self.alpha, self.b,
                                  self.kfun, self.kparam)
        labels, logits = ret
        return labels, logits + 0.5

    def loss(self, Y, P):
        return pvml.hinge_loss(Y, P - 0.5)


@_register_model("svm")
class LinearSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, True)
        self.w = None
        self.b = 0

    def train_step(self, X, Y, steps):
        ret = pvml.svm_train(X, Y, lr=self.lr, lambda_=self.lambda_,
                             steps=steps, init_w=self.w,
                             init_b=self.b)
        self.w, self.b = ret

    def inference(self, X):
        labels, logits = pvml.svm_inference(X, self.w, self.b)
        return labels, logits + 0.5

    def loss(self, Y, P):
        return pvml.hinge_loss(Y, P - 0.5)


@_register_model("multinomial")
class MultinomialLogisticRegressionModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.w = None
        self.b = None

    def train_step(self, X, Y, steps):
        self.w, self.b = pvml.multinomial_logreg_train(
            X, Y, lr=self.lr,
            lambda_=self.lambda_,
            steps=steps, init_w=self.w,
            init_b=self.b)

    def inference(self, X):
        P = pvml.multinomial_logreg_inference(X, self.w, self.b)
        Z = np.argmax(P, 1)
        return Z, P

    def loss(self, Y, P):
        return pvml.cross_entropy(Y, P)


@_register_model("ovo_svm")
class OvoSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.W = None
        self.b = None

    def train_step(self, X, Y, steps):
        ret = pvml.one_vs_one_svm_train(X, Y, lr=self.lr, lambda_=self.lambda_,
                                        steps=steps, init_w=self.W,
                                        init_b=self.b)
        self.W, self.b = ret

    def inference(self, X):
        return pvml.one_vs_one_svm_inference(X, self.W, self.b)


@_register_model("ovr_svm")
class OvrSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.W = None
        self.b = None

    def train_step(self, X, Y, steps):
        ret = pvml.one_vs_rest_svm_train(X, Y, lr=self.lr, lambda_=self.lambda_,
                                         steps=steps, init_w=self.W,
                                         init_b=self.b)
        self.W, self.b = ret

    def inference(self, X):
        return pvml.one_vs_rest_svm_inference(X, self.W, self.b)


@_register_model("ovo_ksvm")
class OvoKSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.Xtrain = None
        self.alpha = None
        self.b = None
        self.kfun = args.kernel
        self.kparam = args.kernel_param

    def train_step(self, X, Y, steps):
        self.Xtrain = X.copy()
        ret = pvml.one_vs_one_ksvm_train(X, Y, self.kfun, self.kparam,
                                         lr=self.lr, lambda_=self.lambda_,
                                         steps=steps, init_alpha=self.alpha,
                                         init_b=self.b)
        self.alpha, self.b = ret

    def inference(self, X):
        return pvml.one_vs_one_ksvm_inference(X, self.Xtrain, self.alpha, self.b,
                                              self.kfun, self.kparam)

@_register_model("ovr_ksvm")
class OvrKSVMModel(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.Xtrain = None
        self.alpha = None
        self.b = None
        self.kfun = args.kernel
        self.kparam = args.kernel_param

    def train_step(self, X, Y, steps):
        self.Xtrain = X.copy()
        ret = pvml.one_vs_rest_ksvm_train(X, Y, self.kfun, self.kparam,
                                          lr=self.lr, lambda_=self.lambda_,
                                          steps=steps, init_alpha=self.alpha,
                                          init_b=self.b)
        self.alpha, self.b = ret

    def inference(self, X):
        return pvml.one_vs_rest_ksvm_inference(X, self.Xtrain, self.alpha, self.b,
                                               self.kfun, self.kparam)


@_register_model("hgda")
class HeteroscedasticGDA(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.means = None
        self.icovs = None
        self.priors = None

    def train_step(self, X, Y, steps):
        ret = pvml.hgda_train(X, Y)
        self.means, self.invcovs, self.priors = ret

    def inference(self, X):
        ret = pvml.hgda_inference(X, self.means, self.invcovs,
                                  self.priors)
        labels, scores = ret
        return labels, scores


@_register_model("ogda")
class OmoscedasticGDA(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.w = None
        self.b = None

    def train_step(self, X, Y, steps):
        self.w, self.b = pvml.ogda_train(X, Y)

    def inference(self, X):
        labels, scores = pvml.ogda_inference(X, self.w, self.b)
        return labels, scores


@_register_model("mindist")
class MinimumDistanceClassifier(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.means = None

    def train_step(self, X, Y, steps):
        self.means = pvml.mindist_train(X, Y)

    def inference(self, X):
        labels, scores = pvml.mindist_inference(X, self.means)
        return labels, scores


@_register_model("categorical_nb")
class CategoricalNaiveBayes(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.probs = None
        self.priors = None

    def train_step(self, X, Y, steps):
        ret = pvml.categorical_naive_bayes_train(X, Y)
        self.probs, self.priors = ret

    def inference(self, X):
        ret = pvml.categorical_naive_bayes_inference(X, self.probs,
                                                     self.priors)
        labels, scores = ret
        return ret


@_register_model("multinomial_nb")
class MultinomialNaiveBayes(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.w = None
        self.b = None

    def train_step(self, X, Y, steps):
        ret = pvml.multinomial_naive_bayes_train(X, Y)
        self.w, self.b = ret

    def inference(self, X):
        ret = pvml.multinomial_naive_bayes_inference(X, self.w,
                                                     self.b)
        labels, scores = ret
        return ret


@_register_model("gaussian_nb")
class GaussianNaiveBayes(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.means = None
        self.vars = None
        self.priors = None

    def train_step(self, X, Y, steps):
        ret = pvml.gaussian_naive_bayes_train(X, Y)
        self.means, self.vars, self.priors = ret

    def inference(self, X):
        ret = pvml.gaussian_naive_bayes_inference(X, self.means,
                                                  self.vars,
                                                  self.priors)
        return ret


@_register_model("classtree")
class ClassificationTree(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.tree = pvml.ClassificationTree()
        self.minsize = args.classtree_minsize
        self.cv = args.classtree_cv
        self.diversity = args.classtree_diversity

    def train_step(self, X, Y, steps):
        self.tree.train(X, Y, minsize=self.minsize,
                        diversity=self.diversity, pruning_cv=self.cv)

    def inference(self, X):
        ret = self.tree.inference(X)
        return ret


@_register_model("perceptron")
class Perceptron(DemoModel):
    def __init__(self, args):
        super().__init__(args, True)
        self.w = None
        self.b = 0

    def train_step(self, X, Y, steps):
        ret = pvml.perceptron_train(X, Y, steps, init_w=self.w,
                                    init_b=self.b)
        self.w, self.b = ret

    def inference(self, X):
        ret = pvml.perceptron_inference(X, self.w, self.b)
        return ret


@_register_model("knn")
class KNN(DemoModel):
    def __init__(self, args):
        super().__init__(args, False, False)
        self.X = None
        self.Y = None
        self.k = args.knn_k

    def train_step(self, X, Y, steps):
        self.X = X.copy()
        self.Y = Y.copy()
        if self.k < 1:
            print("Select K... ", end="", flush=True)
            self.k, acc = pvml.knn_select_k(X, Y)
            print("{} ({:.3f}%)".format(self.k, acc * 100))

    def inference(self, X):
        ret = pvml.knn_inference(X, self.X, self.Y, self.k)
        return ret


@_register_model("kmeans")
class KMeans(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.k = 2
        self.centroids = None

    def train_step(self, X, Y, steps):
        new_k = Y.max() + 1
        if new_k > self.k:
            # If the classes change centroids are reset
            self.centroids = None
            self.k = new_k
        self.centroids = pvml.kmeans_train(X, self.k, steps=steps,
                                           init_centroids=self.centroids)
        self._sort_centroids(X, Y)

    def inference(self, X):
        ret = pvml.kmeans_inference(X, self.centroids)
        return ret

    def _sort_centroids(self, X, Y):
        # K-means labels do not correspond to training labels.  A
        # categorical classifier is used to reorder the centroids to
        # minimize the error.
        P, _ = pvml.kmeans_inference(X, self.centroids)
        probs, priors = pvml.categorical_naive_bayes_train(P[:, None], Y)
        YK = np.arange(self.k)[:, None]
        Q, _ = pvml.categorical_naive_bayes_inference(YK, probs, priors)
        ii = np.argsort(Q)
        self.centroids = self.centroids[ii, :]


@_register_model("mlp")
class MultiLayerPerceptron(DemoModel):
    def __init__(self, args):
        super().__init__(args, False)
        self.net = None
        self.hidden = [int(x) for x in args.mlp_hidden.split(",") if x.strip()]
        self.momentum = args.mlp_momentum
        self.batch = args.mlp_batch

    def train_step(self, X, Y, steps):
        if self.net is None:
            counts = [X.shape[1]] + self.hidden + [Y.max() + 1]
            self.net = pvml.MLP(counts)
        self.net.train(X, Y, lr=self.lr, lambda_=self.lambda_,
                       momentum=self.momentum, steps=steps,
                       batch=self.batch)

    def inference(self, X):
        labels, scores = self.net.inference(X)
        return labels, scores

    def loss(self, Y, P):
        return self.net.loss(Y, P)


def select_features(X, Y, features, class_):
    if features is None and class_ == -1:
        return X, Y
    if features is None:
        features = np.arange(X.shape[1] - 1)
    else:
        features = np.array(list(map(int, features.split(","))))
    data = np.concatenate((X, Y[:, None]), 1)
    X = data[:, features]
    Y = data[:, class_]
    return X, Y


def normalization(X, Xtest, fun):
    if Xtest is None:
        return _NORMALIZATION[fun](X), None
    else:
        return _NORMALIZATION[fun](X, Xtest)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    X, Y = pvml.load_dataset(args.train)
    X, Y = select_features(X, Y, args.features, args.class_)
    if args.test:
        Xtest, Ytest = pvml.load_dataset(args.test)
        Xtest, Ytest = select_features(Xtest, Ytest, args.features,
                                       args.class_)
    else:
        Xtest, Ytest = None, None
    X, Xtest = normalization(X, Xtest, args.normalization)
    model = _MODELS[args.model](args)
    if model.binary:
        Y = (Y > 0).astype(int)
        if Ytest is not None:
            Ytest = (Ytest > 0).astype(int)
    plt.ion()
    model.train(X, Y, Xtest, Ytest, args.steps)
    plt.ioff()
    print("TRAINING COMPLETED")
    plt.show()


if __name__ == "__main__":
    main()
