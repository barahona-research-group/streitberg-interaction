import numpy as np


def xor_4way(dim, length, noise_dim):
    # 4way XOR example where you only have a 4way interaction but no 3way or 2way
    # generate 4 variables
    x1 = np.random.uniform(0, length, dim).reshape(-1, 1)
    x2 = np.random.uniform(0, length, dim).reshape(-1, 1)
    x3 = np.random.uniform(0, length, dim).reshape(-1, 1)
    x4 = np.random.uniform(0, length, dim).reshape(-1, 1)
    x4[noise_dim:] = (x1[noise_dim:] + x2[noise_dim:] + x3[noise_dim:]) % length
    # stack the 4 variables in a list
    X = [x1, x2, x3, x4]
    # add noise
    return X


def normal(d, length, fac=13):
    # factorisations = p12p34, p1p234
    # default is p12p34
    mean = [0, 0, 0, 0]
    cov = [[1, d, 0, 0],
           [d, 1, 0, 0],
           [0, 0, 1, d],
           [0, 0, d, 1]]
    # option for p1p234
    if fac == 13:
        cov = [[1, 0, 0, 0],
               [0, 1, d, d],
               [0, d, 1, d],
               [0, d, d, 1]]
    x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, length).T
    # add white noise
    x1 = (x1 + np.random.normal(0, 0.1, length)).reshape(-1, 1)
    x2 = (x2 + np.random.normal(0, 0.1, length)).reshape(-1, 1)
    x3 = (x3 + np.random.normal(0, 0.1, length)).reshape(-1, 1)
    x4 = (x4 + np.random.normal(0, 0.1, length)).reshape(-1, 1)

    # stack the 4 variables in a list
    X = [x1, x2, x3, x4]
    return X

