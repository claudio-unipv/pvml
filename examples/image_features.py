import numpy as np


def color_histogram(img, bins=64):
    """RGB marginal histgrams of a color image.

    Gray-level images of shape (m, n) are first converted to RGB.
    Histograms are normalized to sum to one.

    Parameters
    ----------
    img : ndarray, shape (m, n) or (m, n, 3)
         input image.
    bins : int
         number of bins in the histograms.

    Returns
    -------
    ndarray, shape (3, nbins)
        marginal histograms.

    """
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)  # Gray to RGB
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    rhist = np.bincount(img[:, :, 0].ravel(), minlength=bins)
    ghist = np.bincount(img[:, :, 1].ravel(), minlength=bins)
    bhist = np.bincount(img[:, :, 2].ravel(), minlength=bins)
    hist = np.stack([rhist, ghist, bhist], 0)
    hist = hist / (img.shape[0] * img.shape[1])
    return hist


def edge_direction_histogram(img, bins=64):
    """Edge direction histogram.

    Color images are first converted to grayscale.  Histograms are
    normalized to sum to one.

    Parameters
    ----------
    img : ndarray, shape (m, n) or (m, n, 3)
         input image.
    bins : int
         number of directions.

    Returns
    -------
    ndarray, shape (3, nbins)
        marginal histograms.

    """
    if img.ndim == 3:
        img = img.sum(2)  # Color to gray
    # Use Sobel's operators
    img = img.astype(float)
    gx = img[:, 2:] - img[:, :-2]
    gx = gx[:-2, :] + 2 * gx[1:-1, :] + gx[2:, :]
    gy = img[2:, :] - img[:-2, :]
    gy = gy[:, :-2] + 2 * gy[:, 1:-1] + gy[:, 2:]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gx, -gy)
    dirs = (bins * angle / (2 * np.pi)).astype(int) % bins
    hist = np.bincount(dirs.ravel(), weights=magnitude.ravel(), minlength=bins)
    hist = hist / max(1e-16, hist.sum())
    return hist


def cooccurrence_matrix(img, bins=8, distance=10):
    """Gray level co-occurrence matrix.

    The matrix represents the distribution of values of neighbor
    pixels.  Color images are first converted to grayscale.  The
    matrix is normalized to sum to one.

    Parameters
    ----------
    img : ndarray, shape (m, n) or (m, n, 3)
         input image.
    bins : int
         number of gray levels.
    distance : int
         distance between neighbor pixels.

    Returns
    -------
    ndarray, shape (nbins, nbins)
        co-occurrence matrix.

    """
    if img.ndim == 3:
        img = img.mean(2)  # Color to gray
    if img.max() > 1:
        img = img / 255.0
    img = (img * (bins - 1)).astype(int)
    # distance pixels below
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)
    # distance pixels to the right
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)
    # the transpose counts pixels above and to the left
    mat += mat.T
    mat = mat / mat.sum()
    return mat


def rgb_cooccurrence_matrix(img, quantization=3, distance=10):
    """Gray level co-occurrence matrix.

    The matrix represents the distribution of colors of adjacent
    pixels.  The matrix is normalized to sum to one.

    Parameters
    ----------
    img : ndarray, shape (m, n, 3)
         input image.
    bins : int
         number of gray levels.
    distance : int
         distance between neighbor pixels.

    Returns
    -------
    ndarray, shape (quantization ** 3, quantization ** 3)
        co-occurrence matrix.

    """
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)  # Gray to RGB
    if img.max() > 1:
        img = img / 255.0
    img = (img * (quantization - 1)).astype(int)
    # Transform colors in indices
    bins = quantization ** 3
    img = (img * np.array([[[1, quantization, quantization ** 2]]])).sum(2)
    # distance pixels below
    mat = _cooccurrence_matrix_dir(img, bins, distance, 0)
    # distance pixels to the right
    mat += _cooccurrence_matrix_dir(img, bins, 0, distance)
    # the transpose counts pixels above and to the left
    mat += mat.T
    mat = mat / mat.sum()
    return mat


def _cooccurrence_matrix_dir(values, bins, di, dj):
    """Helper for the computation of the co-occurrence matrix.

    (di, dj) is the spatial displacement of neighbor pixels.  The
    elements of values must be integers in the range [0, 1, ..., bins)

    """
    m, n = values.shape
    codes = values[:m - di, :n - dj] + bins * values[di:, dj:]
    entries = np.bincount(codes.ravel(), minlength=bins ** 2)
    return entries.reshape(bins, bins)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    for f in sys.argv[1:]:
        img = plt.imread(f)
        h = color_histogram(img)
        np.savetxt(sys.stdout, h.reshape(1, -1), fmt="%.4g")
        print()
        e = edge_direction_histogram(img)
        np.savetxt(sys.stdout, e.reshape(1, -1), fmt="%.4g")
        print()
        m = cooccurrence_matrix(img)
        np.savetxt(sys.stdout, m.reshape(1, -1), fmt="%.4g")
        print()
        m = rgb_cooccurrence_matrix(img)
        np.savetxt(sys.stdout, m.reshape(1, -1), fmt="%.4g")
        print()
