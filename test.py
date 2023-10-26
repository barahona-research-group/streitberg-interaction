import numpy as np

from preprocessing import compute_kernel_mats, centering_kernel_mats
from statistic import compute_streitberg_4way_statistics, compute_dHSIC_statistics, compute_streitberg_5way_statistics, \
    compute_lancaster_4way_statistics, compute_3way_statistics


def permute_K(K, n_perms, seed, perm_var, composite=False):
    """
    this function permutes the kernel matrices of the perm_var
    :param composite: if True, the permutation is applied to the composite subhypotheses
    :param K: list of kernel matrices
    :param n_perms: number of permutations
    :param seed: seed
    :param perm_var: list of variable indices to be permuted,
                    the permutation is the same for all the indices in this list
    :return: K_perms, list of permuted kernel matrices
    """

    K_perms = []
    for seed in np.random.SeedSequence(seed).spawn(n_perms):
        K_perm = np.zeros_like(K)
        n_var = K.shape[0]
        if composite:
            # this is used when composite independence is tested
            rng = np.random.default_rng(seed)
            # variable ordering after permutation
            var_perm = rng.permutation(K.shape[1])
            for i in range(n_var):
                if i in perm_var:
                    K_perm[i, :, :] = K[i, var_perm, var_perm[:, None]]
                else:
                    K_perm[i, :, :] = K[i, :, :]
            K_perms.append(K_perm)
        else:
            # this is used when joint independence is tested
            for d, sub_seed in enumerate(seed.spawn(n_var)):
                rng = np.random.default_rng(sub_seed)
                var_perm = rng.permutation(K.shape[1])
                K_perm[d, :, :] = K[d, var_perm, var_perm[:, None]]
            K_perms.append(K_perm)
    return K_perms


def reject_H0(K0, K_perms, stat_fun, alpha=0.05):
    """
    Approximates the null distribution by permutation. Using Monte Carlo approximation.
    :param K0: kernel matrix of the original data
    :param K_perms: list of kernel matrices of the permuted data
    :param stat_fun: function to compute the test statistic
    :param alpha: significance level
    :return: 1 if H0 is rejected, 0 otherwise
    """
    s0 = stat_fun(K0)
    stats = list(map(stat_fun, K_perms))
    return int((1 + sum(s >= s0 for s in stats)) / (1 + len(stats)) < alpha)


def pairwise_test(X, n_perms, seed=0, alpha=0.05):
    """
    Tests pairwise independence of the data.
    :param X: list of data matrices of shape
    :param n_perms: number of permutations
    :param seed: seed
    :param alpha: significance level
    :return: 1 if H0 is rejected, 0 otherwise
    """
    K = compute_kernel_mats(X)
    K_perms = permute_K(K, n_perms, seed, None)
    reject = reject_H0(K, K_perms, compute_dHSIC_statistics, alpha=alpha)
    return reject


def composite_4way_factorisability_test_dHSIC(X, n_perms, seed=0, alpha=0.05):
    """
    Tests composite 4-way factorisability of the data.
    :param X: list of data matrices of shape
    :param n_perms: number of permutations
    :param seed: seed
    :param alpha: significance level
    :return: 1 if H0 is rejected, 0 otherwise
    """
    K = compute_kernel_mats(X)
    reject7 = 0

    # test1: p1234 = p1p234
    K[1] = K[1] * K[2] * K[3]
    K_perms = permute_K(K[:2], n_perms, seed, None, composite=False)
    reject1 = reject_H0(K[:2], K_perms, compute_dHSIC_statistics, alpha=alpha)
    # do next test only if the first test is rejected
    if reject1:
        # test2: p1234 = p2p134
        K = compute_kernel_mats(X)
        K[0] = K[0] * K[2] * K[3]
        K_perms = permute_K(K[:2], n_perms, seed, None, composite=False)
        reject2 = reject_H0(K[:2], K_perms, compute_dHSIC_statistics, alpha=alpha)
        # do next test only if the second test is rejected
        if reject2:
            # test3: p1234 = p3p124
            K = compute_kernel_mats(X)
            K[3] = K[0] * K[1] * K[3]
            K_perms = permute_K(K[2:], n_perms, seed, None, composite=False)
            reject3 = reject_H0(K[2:], K_perms, compute_dHSIC_statistics, alpha=alpha)
            # do next test only if the third test is rejected
            if reject3:
                # test4: p1234 = p4p123
                K = compute_kernel_mats(X)
                K[2] = K[0] * K[1] * K[2]
                K_perms = permute_K(K[2:], n_perms, seed, None, composite=False)
                reject4 = reject_H0(K[2:], K_perms, compute_dHSIC_statistics, alpha=alpha)
                # do next test only if the fourth test is rejected
                if reject4:
                    # test5: p1234 = p12p34
                    K = compute_kernel_mats(X)
                    K[0] = K[0] * K[1]
                    K[1] = K[2] * K[3]
                    K_perms = permute_K(K[:2], n_perms, seed, None, composite=False)
                    reject5 = reject_H0(K[:2], K_perms, compute_dHSIC_statistics, alpha=alpha)
                    # do next test only if the fifth test is rejected
                    if reject5:
                        # test6: p1234 = p13p24
                        K = compute_kernel_mats(X)
                        K[0] = K[0] * K[2]
                        K[1] = K[1] * K[3]
                        K_perms = permute_K(K[:2], n_perms, seed, None, composite=False)
                        reject6 = reject_H0(K[:2], K_perms, compute_dHSIC_statistics, alpha=alpha)
                        # do next test only if the sixth test is rejected
                        if reject6:
                            # test7: p1234 = p14p23
                            K = compute_kernel_mats(X)
                            K[0] = K[0] * K[3]
                            K[1] = K[1] * K[2]
                            K_perms = permute_K(K[:2], n_perms, seed, None, composite=False)
                            reject7 = reject_H0(K[:2], K_perms, compute_dHSIC_statistics, alpha=alpha)

    return reject7


def composite_4way_factorisability_test_streitberg(X, n_perms, seed, alpha=0.05):
    """
    Tests composite 4-way factorisability of the data.
    :param X: list of data matrices of shape
    :param n_perms: number of permutations
    :param seed: seed
    :param alpha: significance level
    :return: 1 if H0 is rejected, 0 otherwise
    """
    K = compute_kernel_mats(X)
    K = centering_kernel_mats(K)
    reject7 = 0

    # test1: p1234 = p1p234
    K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0], composite=True)
    reject1 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
    # do next test only if the first test is rejected
    if reject1:
        # test2: p1234 = p2p134
        K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[1], composite=True)
        reject2 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
        # do next test only if the second test is rejected
        if reject2:
            # test3: p1234 = p3p124
            K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[2], composite=True)
            reject3 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
            # do next test only if the third test is rejected
            if reject3:
                # test4: p1234 = p4p123
                K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[3], composite=True)
                reject4 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
                # do next test only if the fourth test is rejected
                if reject4:
                    # test5: p1234 = p12p34
                    K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0, 1], composite=True)
                    reject5 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
                    # do next test only if the fifth test is rejected
                    if reject5:
                        # test6: p1234 = p13p24
                        K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0, 2], composite=True)
                        reject6 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
                        # do next test only if the sixth test is rejected
                        if reject6:
                            # test7: p1234 = p14p23
                            K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0, 3], composite=True)
                            reject7 = reject_H0(K, K_perms, compute_streitberg_4way_statistics, alpha=alpha)
                            # if reject7:
                            #     print('Passed test')
    return reject7


def composite_lancaster_4way(X, n_perms, seed, alpha=0.05):
    """
    Composite test for Lancaster's 4-way interaction.
    """
    # compute K
    K = compute_kernel_mats(X)
    # center K
    K = centering_kernel_mats(K)

    reject4 = 0
    # test1: p1234 = p1p234
    K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0], composite=True)
    reject1 = reject_H0(K, K_perms, compute_lancaster_4way_statistics, alpha=alpha)
    # do next test only if the first test is rejected
    if reject1:
        # test2: p1234 = p2p134
        K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[1], composite=True)
        reject2 = reject_H0(K, K_perms, compute_lancaster_4way_statistics, alpha=alpha)
        # do next test only if the second test is rejected
        if reject2:
            # test3: p1234 = p3p124
            K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[2], composite=True)
            reject3 = reject_H0(K, K_perms, compute_lancaster_4way_statistics, alpha=alpha)
            # do next test only if the third test is rejected
            if reject3:
                # test4: p1234 = p4p123
                K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[3], composite=True)
                reject4 = reject_H0(K, K_perms, compute_lancaster_4way_statistics, alpha=alpha)

    return reject4


def composite_3way(X, n_perms, seed, alpha=0.05):
    """
    Composite test for 3-way interaction.
    """
    # compute K
    K = compute_kernel_mats(X)
    # center K
    K = centering_kernel_mats(K)

    reject3 = 0
    # test1: p123 = p12p3
    K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[0], composite=True)
    reject1 = reject_H0(K, K_perms, compute_3way_statistics, alpha=alpha)
    # do next test only if the first test is rejected
    if reject1:
        # test2: p123 = p13p2
        K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[1], composite=True)
        reject2 = reject_H0(K, K_perms, compute_3way_statistics, alpha=alpha)
        # do next test only if the second test is rejected
        if reject2:
            # test3: p123 = p23p1
            K_perms = permute_K(K=K, n_perms=n_perms, seed=seed, perm_var=[2], composite=True)
            reject3 = reject_H0(K, K_perms, compute_3way_statistics, alpha=alpha)

    return reject3
