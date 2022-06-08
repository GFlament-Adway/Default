from process import OU_process
import numpy as np

def get_data(n, T, p, censure_rate = 0.25, kappa=0.02):
    """
    Generate data for T timesteps.

    X contains explainatory variables, 3d matrix of sizes : p*n*T
    betas contains true parameters to be estimated, list of size p
    Times containes the true duration until event, list of size n
    Y containes the frailty, represented as a list of size T

    :param T: Number of times
    :param kappa: Value close to estimation found by D. Duffie
    :param p: Number of parameters
    :param n: Sample size
    :return:
    """
    OU = OU_process(kappa, burn = 10)
    Y = OU.get_OU(T)
    X = [[[np.random.choice([-1,1])*np.random.normal(1, 1) for _ in range(p)] for _ in range(n)] for _ in range(T)]
    betas = [np.random.normal(1, 1) for _ in range(p)]
    eta = np.random.normal(-0.5, 0.1)
    intensities = [[np.exp(-np.sum([betas[j] * X[k][i][j] for j in range(p)]) - eta * Y[k]) for k in range(T)] for i in range(n)]
    t = np.arange(T)
    L = np.array([[np.sum(intensities[k][:i]) for k in range(n)] for i in t]).T
    Times = []
    Cens = []
    for i in range(n):
        U = np.random.uniform(0, 1)
        value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
        idx = np.where(L[i] == value)[0][0]
        time = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
        Times += [min(time, T-1)]
        if min(time, T-1) == T-1:
            Cens += [0]
        else:
            Cens += [1 if np.random.uniform(0, 1) > censure_rate else 0]
    print(np.mean(Times))
    return X, Y, Times, Cens, betas, eta
