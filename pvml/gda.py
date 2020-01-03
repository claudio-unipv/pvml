import numpy as np


#!begin1
def hgda_train(X, Y, priors=None):
    """Train a heteroscedastic GDA classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.
    priors : ndarray, shape (k,)
        Prior probabilities for the classes (if None they get
        estimated from Y).

    Returns
    -------
    means : ndarray, shape (k, n)
         class mean vectors.
    invcovs : ndarray, shape (k, n, n)
         inverse of the class covariance matrices.
    priors : ndarray, shape (k,)
         class prior probabilities.
    """
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    invcovs = np.empty((k, n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
        cov = np.cov(X[indices, :].T)
        invcovs[c, :, :] = np.linalg.inv(cov)
    return (means, invcovs, priors)


def hgda_inference(X, means, invcovs, priors):
    """Heteroscedastic GDA inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    means : ndarray, shape (k, n)
         class mean vectors.
    invcovs : ndarray, shape (k, n, n)
         inverse of the class covariance matrices.
    priors : ndarray, shape (k,)
         class prior probabilities.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    """
    m, n = X.shape
    k = means.shape[0]
    scores = np.empty((m, k))
    for c in range(k):
        det = np.linalg.det(invcovs[c, :, :])
        diff = X - means[None, c, :]
        q = ((diff @ invcovs[c, :, :]) * diff).sum(1)
        scores[:, c] = 0.5 * q - 0.5 * np.log(det) - np.log(priors[c])
    labels = np.argmin(scores, 1)
    return labels
#!end1


#!begin2
def ogda_train(X, Y, priors=None):
    """Train a omoscedastic GDA classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.
    priors : ndarray, shape (k,)
        Prior probabilities for the classes (if None they get
        estimated from Y).

    Returns
    -------
    W : ndarray, shape (n, k)
         weight vectors, each row representing a different class.
    b : ndarray, shape (k,)
         vector of biases.
    """
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    cov = np.zeros((n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
        cov += priors[c] * np.cov(X[indices, :].T)
    icov = np.linalg.inv(cov)
    W = -(icov @ means.T)
    q = ((means @ icov) * means).sum(1)
    b = 0.5 * q - np.log(priors)
    return (W, b)


def ogda_inference(X, W, b):
    """Omoscedastic GDA inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    W : ndarray, shape (n, k)
         weight vectors, each row representing a different class.
    b : ndarray, shape (k,)
         vector of biases.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    """
    scores = X @ W + b.T
    labels = np.argmin(scores, 1)
    return labels
#!end2


#!begin3
def mindist_train(X, Y):
    """Train a minimum distance classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with values in {0, ..., k - 1}.

    Returns
    -------
    means : ndarray, shape (k, n)
         class mean vectors.
    """
    k = Y.max() + 1
    n = X.shape[1]
    means = np.empty((k, n))
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
    return (means)


def mindist_inference(X, means):
    """Minimum distance classifier inference.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    means : ndarray, shape (k, n)
         class mean vectors.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    """
    sqdists = ((X[:, None, :] - means[None, :, :]) ** 2).sum(2)
    labels = np.argmin(sqdists, 1)
    return labels
#!end3


if __name__ == "__main__":
    import demo

    class Demo(demo.Demo):
        def train(self, X, Y):
            if self.args.model[0] == "h":
                self.m, self.ic, self.priors = hgda_train(X, Y)
            elif self.args.model[0] == "o":
                self.W, self.b = ogda_train(X, Y)
            else:
                self.m = mindist_train(X, Y)

        def inference(self, X):
            if self.args.model[0] == "h":
                return hgda_inference(X, self.m, self.ic, self.priors)
            elif self.args.model[0] == "o":
                return ogda_inference(X, self.W, self.b)
            else:
                return mindist_inference(X, self.m)

    app = Demo()
    app.parser.add_argument("-m", "--model",
                            choices=["heteroscedastic",
                                     "omoscedastic", "mindist", "h", "o", "m"],
                            default="heteroscedastic",
                            help="Statistical model")
    app.run()

    def icov_data(ic):
        cov = np.linalg.inv(ic)
        w, v = np.linalg.eig(cov)
        print("Evals", w)
        print("Evectors", v)
        scaled = v * np.sqrt(w)
        print("Scaled", scaled)
        print("Determinant", np.linalg.det(cov))
    if app.args.model[0] == "h":
        print("Means")
        print(app.m)
        print("Priors", app.priors)
        for c in app.ic:
            icov_data(c)
        cov = np.zeros_like(c)
        for p, c in zip(app.priors, app.ic):
            cov += p * np.linalg.inv(c)
        print("Global covariance")
        print(cov)
        icov_data(np.linalg.inv(cov))
    elif app.args.model[0] == "o":
        print("W")
        print(app.W)
        print("b")
        print(app.b)
