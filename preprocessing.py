import warnings
import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def width(Z):
    # Computes the median heuristic for the kernel bandwidth
    dist_mat = pairwise_distances(Z, metric='euclidean')
    width_Z = np.median(dist_mat[dist_mat > 0])
    return width_Z


def compute_kernel_n(X):
    # data preparation for individual variable matrices
    """
    Args:
        X: data matrix of shape (n, T)
    """
    return pairwise_kernels(X, metric='rbf', gamma=0.5 / (width(X) ** 2))


def centering_kernel(k):
    # centering a kernel matrix
    """
    Args:
        k: kernel matrix of shape (n, n)
    """
    n = k.shape[0]
    H = np.eye(n) - 1 / n * np.ones(n)
    k_centered = H @ k @ H
    return k_centered


def centering_kernel_mats(K):
    k_list = [K[i] for i in range(K.shape[0])]
    return np.stack(list(map(centering_kernel, k_list)))


def compute_kernel_mats(X_list):
    return np.stack(list(map(compute_kernel_n, X_list)))
