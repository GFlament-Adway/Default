from process import OU_process
import numpy as np

def get_data(n, T, censure_rate = 0.25, kappa=0.02):
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
    OU = OU_process(kappa, burn = 100)
    Y = OU.get_OU(T)
    betas = [-1, -1.2, -0.65, -0.25, 1.55]
    p = len(betas)
    X = [[[1] + [np.random.uniform(10, 25) for _ in range(p - 1)] for _ in range(n)] for _ in range(T)]
    #betas = [np.random.choice([-1, 1], p = [0.8, 0.2])*np.random.normal(0.9, 0.2) for _ in range(p+1)]
    #data = [[np.sum([betas[j] * X[k][i][j] for j in range(p)]) for k in range(T)] for i in range(n)]


    eta = 0.12 #As in D. Duffie
    intensities = [[np.exp(np.sum([betas[j] * X[k][i][j] for j in range(p)]) + eta * Y[k]) for k in range(T)] for i in range(n)]

    t = np.arange(T)
    C = [np.random.exponential(20) for _ in range(n)]
    L = np.array([[np.sum(intensities[k][:i]) for k in range(n)] for i in t]).T
    Times = []
    Cens = []
    for i in range(n):
        U = np.random.uniform(0, 1)
        value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
        idx = np.where(L[i] == value)[0][0]
        time = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
        Times += [min(time, T-1)]
        print(time, C[i], 1 if (time > C[i] or time > T) else 0)
        Cens += [1 if (time > C[i] or time > T) else 0]
    return X, Y, Times, Cens, betas, eta

if __name__ == "__main__":
    X,Y,Times,Cens, betas, eta  = get_data(400, 20)
    print(np.sum(Cens)/len(Cens))
    print(np.mean(Times))
    print(len(Cens))
    print(len(Times))