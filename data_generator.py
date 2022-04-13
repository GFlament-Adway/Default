import numpy as np

def generate_default_times(intensity=0.2, censure_intensity=0.08, n_obs=100):
    """

    :param intensity:
    :param censure_intensity:
    :param n_obs:
    :return:
    """
    times = np.random.exponential(1 / intensity, n_obs)
    censures = np.random.exponential(1 / censure_intensity, n_obs)
    return [min(times[k], censures[k]) for k in range(n_obs)], [1 if times[k] < censures[k] else 0 for k in
                                                                range(n_obs)]



def generate_default(n_indiv=5000, n_indiv_test = 2500, n_params=10):
    """
    Generate data
    :param n_indiv: Size of train sample
    :param n_indiv_test: Size of test sample
    :param n_params: number of features
    :return: a dataset
    """
    X = [list(np.random.normal(0, 0.2, n_params + 1)) for _ in range(n_indiv)]
    X_test = [list(np.random.normal(0, 0.2, n_params + 1)) for _ in range(n_indiv_test)]
    betas = np.random.uniform(-1, 1, int(n_params * (n_params + 1) / 2))

    Y = [betas[0] +
         np.sum([betas[k + 1]*X[i][k] for k in range(n_params)])
         for i in range(n_indiv)]
    Y_test = [betas[0] +
              np.sum([betas[k + 1]*X_test[i][k] for k in range(n_params)])
              for i in range(n_indiv_test)]
    Y = [1 / (1 + np.exp(-Y[i])) for i in range(n_indiv)]
    Y_test = [1 / (1 + np.exp(-Y_test[i])) for i in range(n_indiv_test)]
    med = np.median(Y)
    med_test = np.median(Y_test)
    Y = [1 if Y[i] > med else 0 for i in range(n_indiv)]
    Y_test = [1 if Y_test[i] > med_test else 0 for i in range(n_indiv_test)]

    return Y, X, X_test, Y_test, betas


def generate_default_hurlin(n_indiv=5000, n_indiv_test=2500, n_params=10, non_linear=True):
    """
    Generate data as in Dumitrescu, Hurlin.
    :param n_indiv: Size of train sample
    :param n_indiv_test: Size of test sample
    :param n_params: number of features
    :return: a dataset
    """

    X = [list(np.random.normal(0, 1, n_params + 1)) for _ in range(n_indiv)]
    X_test = [list(np.random.normal(0, 1, n_params + 1)) for _ in range(n_indiv_test)]
    lower_bonds = [sorted([X[i][k] for i in range(n_indiv)])[int(0.1 * n_indiv)] for k in range(n_params)]
    upper_bonds = [sorted([X[i][k] for i in range(n_indiv)])[int(0.9 * n_indiv)] for k in range(n_params)]
    betas = np.random.uniform(-1, 1, int(n_params * (n_params + 1) / 2))
    gammas = [np.random.uniform(lower_bonds[k], upper_bonds[k], 1) for k in range(n_params)]
    deltas = [np.random.uniform(lower_bonds[k], upper_bonds[k], 1) for k in range(n_params)]
    Y = [betas[0] +
         np.sum([betas[k + 1] if X[i][k] < gammas[k] else 0 for k in range(n_params)]) +
         np.sum([betas[k + j] if X[i][k] < deltas[k] and X[i][j] < deltas[j] else 0 for k in range(n_params) for j in
                 range(k, n_params)])
         for i in range(n_indiv)]
    Y_test = [betas[0] +
              np.sum([betas[k + 1] if X_test[i][k] < gammas[k] else 0 for k in range(n_params)]) +
              np.sum(
                  [betas[k + j] if X_test[i][k] < deltas[k] and X_test[i][j] < deltas[j] else 0 for k in range(n_params)
                   for j in
                   range(k, n_params)])
              for i in range(n_indiv_test)]
    if non_linear:
        for i in range(n_indiv):
            X[i] += [X[i][k] ** 2 for k in range(n_params)] + [X[i][k] * X[i][j] for k in range(n_params) for j in
                                                               range(k, n_params)]
        for i in range(n_indiv_test):
            X_test[i] += [X_test[i][k] ** 2 for k in range(n_params)] + [X_test[i][k] * X_test[i][j] for k in
                                                                         range(n_params) for j in
                                                                         range(k, n_params)]
    Y = [1 / (1 + np.exp(-Y[i])) for i in range(n_indiv)]
    Y_test = [1 / (1 + np.exp(-Y_test[i])) for i in range(n_indiv_test)]
    med = np.median(Y)
    med_test = np.median(Y_test)
    Y = [1 if Y[i] > med else 0 for i in range(n_indiv)]
    Y_test = [1 if Y_test[i] > med_test else 0 for i in range(n_indiv_test)]

    return Y, X, X_test, Y_test, betas, gammas, deltas

if __name__ == "__main__":
    n_indiv = 500
    n_indiv_test = 100
    param = 5
    Y, X, X_test, Y_test, betas, gammas,deltas  = generate_default_hurlin(n_indiv=n_indiv, n_indiv_test=n_indiv_test,
                                                                   n_params=param,
                                                                  non_linear=False)
    clf = DecisionTreeClassifier().fit(X,Y)
    X, X_test = pltr(X, Y, X_test)