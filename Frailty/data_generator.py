from process import OU_process
import numpy as np


def get_data(n, T, p, censure_rate = 0.25):
    """
    Generate data for T timesteps.

    X contains explainatory variables, 3d matrix of sizes : p*n*T
    betas contains true parameters to be estimated, list of size p
    Times containes the true duration until event, list of size n
    Y containes the frailty, represented as a list of size T

    :param T: Number of times
    :param p: Number of parameters
    :param n: Sample size
    :return:
    """
    alpha = 0.05
    beta = 0.1
    gamma = 0

    OU = OU_process(alpha, beta, gamma)
    Y = OU.get_OU(T)
    X = [[[np.random.normal(0, 0.2) for _ in range(p)] for _ in range(n)] for _ in range(T)]
    betas = [np.random.normal(0, 1) for _ in range(p)]
    intensities = [[np.exp(-np.sum([betas[j] * X[k][i][j] for j in range(p)])) for k in range(T)]for i in range(n)]
    t = np.arange(T)
    L = np.array([[np.sum(intensities[k][:i]) for k in range(n)] for i in t]).T
    Times = []
    Cens = []
    for i in range(n):
        U = np.random.uniform(0,1)
        value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
        idx = np.where(L[i] == value)[0][0]
        t = ((-np.log(1 - U) - value) + intensities[i][idx] * idx)/intensities[i][idx]
        Times += [t]
        Cens += [1 if np.random.uniform(0, 1) > censure_rate else 0]
    return X, Y, Times, Cens, betas
