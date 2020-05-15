import numpy as np
import urllib.request
import gzip
import sys


_FACTORIES = {}


def load_dataset(datasetname):
    """Load or generate a data set given its name.

    If the name is unknown then it interprets it as a path to a data
    file.  The file should be a text file with one vector per row and
    with elements separated by spaces).  The last element is used as
    class label.

    Parameters
    ----------
    datasetname : str
         name of the dataset or path to a file

    Returns
    -------
    ndarray, shape (m, n)
        features.
    ndarray, shape (m,)
        class labels.

    """
    try:
        f = _FACTORIES[datasetname]
        return f()
    except KeyError:
        pass
    data = np.loadtxt(datasetname)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    return X, Y


def _register(name):
    def wrapper(f):
        _FACTORIES[name] = f
        return f
    return wrapper


@_register("gaussian")
def _gaussian_dataset():
    """Two Gaussians."""
    sz = 100
    Y = np.arange(sz) % 2
    X = np.random.randn(sz, 2)
    X[:, 0] += 3 * Y
    return X, Y


@_register("gaussian4")
def _gaussian4_dataset():
    """Four gaussians."""
    sz = 100
    Y = np.arange(sz) % 4
    X = np.random.randn(sz, 2)
    X[:, 0] += 4 * (Y // 2)
    X[:, 1] += 4 * (Y % 2)
    return X, Y


@_register("xor")
def _xor_dataset():
    """Exclusive OR configuration."""
    sz = 100
    X = np.random.randn(sz, 2) / 10
    X[(sz // 2):, 0] += 1
    X[(sz // 4):(3 * sz // 4), 1] += 1
    Y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)
    return X, Y


@_register("rings")
def _rings_dataset():
    """Two concentric rings."""
    sz = 100
    Y = np.arange(sz) % 2
    t = 2 * np.pi * np.random.rand(sz)
    r = Y + 1
    X = np.vstack([r * np.cos(t), r * np.sin(t)]).T
    X += 0.1 * np.random.randn(*X.shape)
    return X, Y


@_register("moons")
def _moons_dataset():
    """Two non-linearly separable moons."""
    sz = 100
    Y = np.arange(sz) % 2
    t = np.linspace(0, np.pi, sz)
    X = np.vstack([np.cos(t + Y * np.pi), np.sin(t + Y * np.pi)]).T
    X += np.outer(Y, np.array([1, 0.3])) + 0.1 * np.random.randn(*X.shape)
    return X, Y


@_register("swissroll")
def _swissroll_dataset():
    """Interwined spirals."""
    sz = 100
    Y = np.arange(sz) % 2
    t = np.linspace(0, 4 * np.pi, sz)
    X = t[:, None] * np.vstack([np.cos(t + Y * np.pi),
                                np.sin(t + Y * np.pi)]).T
    X += 0.2 * np.random.randn(*X.shape)
    return X, Y


@_register("categorical")
def _categorical_dataset():
    """Discrete features."""
    sz = 100
    t = np.arange(sz) % 36
    X = np.vstack([t % 6, t // 6]).T
    Y = ((t % 3) == 2).astype(int)
    s = (np.random.rand(sz) > 0.9)
    Y[s] = 1 - Y[s]
    X = X + np.random.random(X.shape) * 0.05
    return X, Y


@_register("yinyang")
def _yinyang_dataset():
    """Yin and yang shape."""
    sz = 200
    t = 2 * np.pi * np.random.rand(sz)
    r = 6 * (np.random.rand(sz) ** 0.5)
    X = np.vstack([r * np.cos(t), r * np.sin(t)]).T
    r1 = ((X - np.array([[0, 3]])) ** 2).sum(1)
    r2 = ((X - np.array([[0, -3]])) ** 2).sum(1)

    Y = np.logical_and(np.logical_or(r1 < 9, X[:, 0] > 0),
                       np.logical_not(r2 < 9))
    Y2 = np.logical_or(r1 < 1.5 ** 2, r2 < 1.5 ** 2)
    Y = np.logical_xor(Y, Y2).astype(int)
    return X, Y


@_register("iris")
def _iris_dataset():
    """The famous 'iris' dataset.

    This version provides only two features: the sepal length and the
    sepal width.

    The three classes are:
      0 => Iris setosa
      1 => Iris versicolor
      2 => Iris virginica

    """
    data = _IRIS_TXT.split()
    x1 = list(map(float, data[0::3]))
    x2 = list(map(float, data[1::3]))
    X = np.stack((np.array(x1), np.array(x2)), 1)
    Y = np.array(list(map(int, data[2::3])))
    return X, Y


_IRIS_TXT = """5.1 3.5 0 4.9 3 0 4.7 3.2 0 4.6 3.1 0 5 3.6 0 5.4 3.9 0 4.6 3.4 0 5
3.4 0 4.4 2.9 0 4.9 3.1 0 5.4 3.7 0 4.8 3.4 0 4.8 3 0 4.3 3 0 5.8 4 0
5.7 4.4 0 5.4 3.9 0 5.1 3.5 0 5.7 3.8 0 5.1 3.8 0 5.4 3.4 0 5.1 3.7 0
4.6 3.6 0 5.1 3.3 0 4.8 3.4 0 5 3 0 5 3.4 0 5.2 3.5 0 5.2 3.4 0 4.7
3.2 0 4.8 3.1 0 5.4 3.4 0 5.2 4.1 0 5.5 4.2 0 4.9 3.1 0 5 3.2 0 5.5
3.5 0 4.9 3.1 0 4.4 3 0 5.1 3.4 0 5 3.5 0 4.5 2.3 0 4.4 3.2 0 5 3.5 0
5.1 3.8 0 4.8 3 0 5.1 3.8 0 4.6 3.2 0 5.3 3.7 0 5 3.3 0 7 3.2 1 6.4
3.2 1 6.9 3.1 1 5.5 2.3 1 6.5 2.8 1 5.7 2.8 1 6.3 3.3 1 4.9 2.4 1 6.6
2.9 1 5.2 2.7 1 5 2 1 5.9 3 1 6 2.2 1 6.1 2.9 1 5.6 2.9 1 6.7 3.1 1
5.6 3 1 5.8 2.7 1 6.2 2.2 1 5.6 2.5 1 5.9 3.2 1 6.1 2.8 1 6.3 2.5 1
6.1 2.8 1 6.4 2.9 1 6.6 3 1 6.8 2.8 1 6.7 3 1 6 2.9 1 5.7 2.6 1 5.5
2.4 1 5.5 2.4 1 5.8 2.7 1 6 2.7 1 5.4 3 1 6 3.4 1 6.7 3.1 1 6.3 2.3 1
5.6 3 1 5.5 2.5 1 5.5 2.6 1 6.1 3 1 5.8 2.6 1 5 2.3 1 5.6 2.7 1 5.7 3
1 5.7 2.9 1 6.2 2.9 1 5.1 2.5 1 5.7 2.8 1 6.3 3.3 2 5.8 2.7 2 7.1 3 2
6.3 2.9 2 6.5 3 2 7.6 3 2 4.9 2.5 2 7.3 2.9 2 6.7 2.5 2 7.2 3.6 2 6.5
3.2 2 6.4 2.7 2 6.8 3 2 5.7 2.5 2 5.8 2.8 2 6.4 3.2 2 6.5 3 2 7.7 3.8
2 7.7 2.6 2 6 2.2 2 6.9 3.2 2 5.6 2.8 2 7.7 2.8 2 6.3 2.7 2 6.7 3.3 2
7.2 3.2 2 6.2 2.8 2 6.1 3 2 6.4 2.8 2 7.2 3 2 7.4 2.8 2 7.9 3.8 2 6.4
2.8 2 6.3 2.8 2 6.1 2.6 2 7.7 3 2 6.3 3.4 2 6.4 3.1 2 6 3 2 6.9 3.1 2
6.7 3.1 2 6.9 3.1 2 5.8 2.7 2 6.8 3.2 2 6.7 3.3 2 6.7 3 2 6.3 2.5 2
6.5 3 2 6.2 3.4 2 5.9 3 2 """


def _load_mnist_set(filename, features_url, labels_url):
    try:
        data = np.load(filename, allow_pickle=True)
        return data["X"], data["Y"]
    except FileNotFoundError:
        pass

    print("Downloading the data set...", file=sys.stderr)
    with urllib.request.urlopen(features_url) as remote:
        with gzip.open(remote) as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            X = data[16:].reshape((-1, (28 * 28))).astype(np.float32) / 255.0
    with urllib.request.urlopen(labels_url) as remote:
        with gzip.open(remote) as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            Y = data[8:].astype(np.int32)

    try:
        np.savez_compressed(filename, X=X, Y=Y)
        print("Dataset saved to '{}'".format(filename), file=sys.stderr)
    except Exception as e:
        print("Failed to save the data set to '{}' ({})".format(filename, e))
    return X, Y


_MNIST_URL = "http://yann.lecun.com/exdb/mnist/"


@_register("mnist_train")
def _mnist_train_dataset():
    """MNIST dataset for handwritten digits recognition (training data).

    The data is downloaded from http://yann.lecun.com/exdb/mnist/
    and saved to the file 'mnist_train.npz' to avoid multiple downloads.
    """
    X, Y = _load_mnist_set("mnist_train.npz",
                           _MNIST_URL + "train-images-idx3-ubyte.gz",
                           _MNIST_URL + "train-labels-idx1-ubyte.gz")
    return X, Y


@_register("mnist_test")
def _mnist_test_dataset():
    """MNIST dataset for handwritten digits recognition (test data).

    The data is downloaded from http://yann.lecun.com/exdb/mnist/
    and saved to the file 'mnist_test.npz' to avoid multiple downloads.
    """
    X, Y = _load_mnist_set("mnist_test.npz",
                           _MNIST_URL + "t10k-images-idx3-ubyte.gz",
                           _MNIST_URL + "t10k-labels-idx1-ubyte.gz")
    return X, Y


_FASHION_MNIST_URL = ("http://fashion-mnist.s3-website.eu-central-1" +
                      ".amazonaws.com/")


@_register("fashion_mnist_train")
def _fashion_mnist_train_dataset():
    """MNIST dataset for handwritten digits recognition (training data).

    The data is downloaded from http://yann.lecun.com/exdb/mnist/
    and saved to the file 'mnist_train.npz' to avoid multiple downloads.
    """
    X, Y = _load_mnist_set("fashion_mnist_train.npz",
                           _FASHION_MNIST_URL + "train-images-idx3-ubyte.gz",
                           _FASHION_MNIST_URL + "train-labels-idx1-ubyte.gz")
    return X, Y


@_register("fashion_mnist_test")
def _fashion_mnist_test_dataset():
    """MNIST dataset for handwritten digits recognition (test data).

    The data is downloaded from http://yann.lecun.com/exdb/mnist/
    and saved to the file 'mnist_test.npz' to avoid multiple downloads.
    """
    X, Y = _load_mnist_set("fashion_mnist_test.npz",
                           _FASHION_MNIST_URL + "t10k-images-idx3-ubyte.gz",
                           _FASHION_MNIST_URL + "t10k-labels-idx1-ubyte.gz")
    return X, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("No dataset given")
        print("Available datasets are:")
        for k, v in sorted(_FACTORIES.items()):
            txt = v.__doc__.splitlines()[0]
            print("{:20s}\t{}".format("'" + k + "'", txt))
        sys.exit()

    try:
        X, Y = load_dataset(sys.argv[1])
    except Exception as e:
        print("Error loading the data set ({})".format(e))
        sys.exit()
    m, n = X.shape
    print("Loaded {} samples with {} features".format(m, n))
    print()
    for k in range(Y.max() + 1):
        count = sum(Y == k)
        print("{} samples of class {}".format(count, k))
    if n == 2:
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
        plt.show()
    if len(sys.argv) > 2:
        np.savetxt(sys.argv[2],
                   np.concatenate([X, Y[:, None]], 1),
                   fmt=(["%.4f"] * X.shape[1]) + ["%d"])
