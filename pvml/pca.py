import numpy as np


def pca(X, Xtest=None, mincomponents=1, retvar=0.95):
    """Principal Component Analysis.

    Perform PCA dimensionality reduction.  The number of output
    components is the maximum between mincomponents and those required
    to ensure that at least a fraction retvar of the original variance
    is retained.

    Features are linearly transformed to have zero mean and null
    covariances.  Test features, when given, are processed using the
    transform estimated on X.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         test features (one row per feature vector).
    mincomponents : int
         minimum number of output components.
    retvar : float
         minimum fraction of total variance retained in the output
         components. 

    Returns
    -------
    ndarray, shape (m, output_n)
        normalized features.
    ndarray, shape (mtest, output_n)
        normalized test features (returned only when Xtest is not None).

    """
    # Compute the moments
    mu = X.mean(0, keepdims=True)
    sigma = np.cov(X.T)
    # Compute and sort the eigenvalues
    evals, evecs = np.linalg.eig(sigma)
    order = np.argsort(-evals)
    evals = evals[order]
    # Determine the components to retain
    k = 1 + (np.cumsum(evals) >= retvar *  evals.sum()).nonzero()[0][0]
    print(k)
    k = max(k, mincomponents)
    w = evecs[:, order[:k]]  # 1e-15 avoids div. by zero
    # Transform the data
    X = (X - mu) @ w
    print(X.mean())
    print(np.cov(X.T))
    if Xtest is None:
        return X
    Xtest = (Xtest - mu) @ w
    return X, Xtest
