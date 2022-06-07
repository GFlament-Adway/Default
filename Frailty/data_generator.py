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
    alpha = 0.01
    beta = 1
    gamma = 0
    OU = OU_process(alpha, beta, gamma)
    Y = OU.get_OU(T)
    X = [[np.random.choice([-1,1])*[np.random.normal(1, 1) for _ in range(p)] for _ in range(n)] for _ in range(T)]
    betas = [np.random.normal(1, 1) for _ in range(p)]
    eta = np.random.normal(-0.5, 0.1)
    print(np.shape(betas), p)
    print(np.shape(Y), T)
    print(np.shape(X), n)
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

def get_data_v2(n, T, p, censure_rate = 0.25):
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
    alpha = 0.01
    beta = 1
    gamma = 0
    import matplotlib.pyplot as plt
    OU = OU_process(alpha, beta, gamma)
    Y = OU.get_OU(T)
    plt.figure()
    plt.plot(Y)
    plt.draw()
    X = [[[np.random.normal(1, 0.1) for _ in range(p)] for _ in range(n)] for _ in range(T)]
    betas = [np.random.normal(3, 1) for _ in range(p)]
    eta = np.random.normal(-0.5, 0.1)
    intensities = [
        [np.exp(-np.sum([betas[j] * X[k][i][j] for j in range(p)]) - eta * Y[k]) for k in range(T)] for i in
        range(n)]
    t = np.arange(T)
    L = np.array([[np.sum(intensities[k][:i]) for k in range(n)] for i in t]).T

    Times = []
    Cens = []
    for ti in range(T):
        if ti == 0:
            for i in range(n):
                U = np.random.uniform(0, 1)
                value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
                idx = np.where(L[i] == value)[0][0]
                time = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
                Times += [(ti, min(time, T - 1), i)]
                Cens += [1 if np.random.uniform(0, 1) > censure_rate else 0]
        else :
            n_defaults = np.sum([time[1] > ti for time in Times])
            id_defaults = [time[2] for time in Times if time[1] > ti]
            print(n_defaults)
            for i in id_defaults:
                U = np.random.uniform(0, 1)
                L = np.array([[np.sum(intensities[k][ti:j]) for k in range(n_defaults)] for j in t]).T
                value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
                idx = np.where(L[i] == value)[0][0]
                time = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
                Times += [(ti, min(time, T - 1), i)]
                Cens += [1 if np.random.uniform(0, 1) > censure_rate else 0]
    print(len(Times))
    plt.figure()
    plt.hist([time[1] for time in Times], bins=20)
    plt.show()
    return X, Y, Times, Cens, betas, eta

if __name__ == "__main__":
    get_data_v2(100, 20, 3)