import numpy as np


def compute_dHSIC_statistics(k_list):
    """
    Computes the dHSIC statistic
    :param k_list: list of kernel matrices for each variable
    :return: single value for dHSIC test statistic
    """

    n_variables = len(k_list)
    n_samples = k_list[0].shape[0]

    term1 = k_list[0]
    term2 = np.sum(k_list[0]) / (n_samples ** 2)
    term3 = 2 * np.sum(k_list[0], axis=0) / (n_samples ** 2)
    for j in range(1, n_variables):
        term1 = term1 * k_list[j]
        term2 = term2 * np.sum(k_list[j]) / (n_samples ** 2)
        term3 = term3 * np.sum(k_list[j], axis=0) / n_samples

    term1_sum = np.sum(term1)
    term3_sum = np.sum(term3)
    dHSIC = term1_sum / (n_samples ** 2) + term2 - term3_sum
    return dHSIC


def compute_3way_statistics(k_list):
    """
    Computes the 3-way Lancaster/Streitberg test statistic
    :param k_list: list of kernel matrices for each variable
    :return: single value for 3-way interaction test statistic
    """
    K = k_list[0]
    L = k_list[1]
    M = k_list[2]
    m = np.shape(K)[0]
    H = np.eye(m) - 1 / m * np.ones(m)
    Kc = H @ K @ H
    Lc = H @ L @ H
    Mc = H @ M @ H
    statMatrix = Kc * Lc * Mc
    threeway = 1 / (m ** 2) * np.sum(statMatrix)
    return threeway


def compute_lancaster_4way_statistics(K):
    """
    Computes the 4-way Lancaster test statistic
    :param K: list of kernel matrices for each variable
    :return: single value for 4-way Lancaster test statistic
    """
    n = K[0].shape[0]
    statMatrix = np.einsum('ij,ij,ij,ij->', K[0], K[1], K[2], K[3])
    stat = 1 / (n ** 2) * np.sum(statMatrix)
    return stat


def compute_streitberg_4way_statistics(K):
    """
    Computes the 4-way Streitberg test statistic
    :param K: list of kernel matrices for each variable
    :return: single value for 4-way Streitberg test statistic
    """
    n = K[0].shape[0]
    # p1234,p1234
    ip1 = np.einsum('ij,ij,ij,ij->', K[0], K[1], K[2], K[3])
    # p1234, p12p34
    ip2 = np.einsum('ij,ij,ik,ik->', K[0], K[1], K[2], K[3])
    # p1234, p13p24
    ip3 = np.einsum('ij,ij,ik,ik->', K[0], K[2], K[1], K[3])
    # p1234, p14p23
    ip4 = np.einsum('ij,ij,ik,ik->', K[0], K[3], K[1], K[2])
    # p12p34, p12p34
    ip5 = np.einsum('ij,ij,kl,kl->', K[0], K[1], K[2], K[3])
    # p12p34, p13p24
    ip6 = np.einsum('ij,il,kj,kl->', K[0], K[1], K[2], K[3])
    # p12p34, p14p23
    ip7 = np.einsum('ij,il,kl,kj->', K[0], K[1], K[2], K[3])
    # p13p24, p13p24
    ip8 = np.einsum('ij,ij,kl,kl->', K[0], K[2], K[1], K[3])
    # p13p24, p14p23
    ip9 = np.einsum('ij,il,kl,kj->', K[0], K[2], K[1], K[3])
    # p14p23, p14p23
    ip10 = np.einsum('ij,ij,kl,kl->', K[0], K[3], K[1], K[2])
    # sum up
    stat = 1 / (n ** 4) * (n ** 2 * ip1 - 2 * n * (ip2 + ip3 + ip4)
                           + ip5 + ip8 + ip10 + 2 * (ip6 + ip7 + ip9))
    return stat
