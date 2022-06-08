import numpy as np
import optimizer
from scipy.optimize import minimize


class Frailty():
    def __init__(self, X, Times, Cens, frailty, betas=None):
        self.last_draw = 0
        self.X = X
        self.Times = Times
        self.T = len(X)
        self.p = len(X[0][0])
        self.n = len(X[0])
        self.C = Cens
        self.Y = frailty
        if betas is None:
            self.betas = [0 for _ in range(self.p)]
        else:
            self.betas = betas
        self.eta = 0.05

    def draw(self):
        self.last_draw = np.random.normal(mean=self.last_draw, scale=2)
        return self.last_draw

    def likelihood(self, param, *args):
        """
        See equation 6.4 from D. Duffie
        :param Y:
        :return:
        """
        if args[0] == "beta":
            intensities = [
                [np.exp(-np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) - self.eta * args[1][k]) for k in
                 range(int(self.Times[i]) + 1)] for
                i in range(self.n)]

        elif args[0] == "eta":
            intensities = [
                [np.exp(-np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) - param * args[1][k]) for k in
                 range(int(self.Times[i]) + 1)] for
                i in range(self.n)]

        log_likelihood = 0
        for i in range(self.n):
            log_likelihood += (1 - self.C[i]) * (
                    -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                    float(self.Times[i]) - int(self.Times[i]))) + \
                              self.C[i] * (np.log(intensities[i][int(self.Times[i])]) - np.sum(
                intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                                                   self.Times[i] - int(self.Times[i])))
        return -log_likelihood

    def fit(self, method=None, frailty=True):
        if frailty:
            self.parameters = minimize(self.likelihood, self.eta, args=("eta", self.Y), method=method)
            self.eta = self.parameters["x"]
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y), method=method)
            self.betas = self.parameters["x"]
        else:
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y), method=method)
            self.betas = self.parameters["x"]
        self.log_likelihood = -self.parameters["fun"]

    def draw(self):
        for k in range(self.T):
            y_k = np.random.normal(self.Y[k], 2)
            new_frailty = [y if i != k else y_k for i, y in enumerate(self.Y)]
            new_like = self.likelihood(self.eta, "eta", new_frailty)
            old_like = self.likelihood(self.eta, "eta", self.Y)
            U = np.random.uniform()
            acceptance = min(np.exp(-new_like) / np.exp(-old_like), 1)
            if U < acceptance:
                self.Y[k] = y_k


if __name__ == "__main__":
    np.random.seed(13)
    from data_generator import get_data
    import matplotlib.pyplot as plt

    X, Y, Times, Cens, betas, eta = get_data(100, 20, 3, censure_rate=0.1)
    print(len(Times))
    print("Parameters to estimate : ", betas, eta)
    frailty = [0 for _ in range(len(Times))]
    no_frailty_model = Frailty(X, Times, Cens, frailty)
    print("#################### First step of Duffie ################")
    print("############ Estimating betas without frailty ############""")
    no_frailty_model.fit(frailty=False)
    print(no_frailty_model.betas)
    print("############ Second step, estimating eta and frailty ##################")
    frailty = [np.random.normal(0, 2) for _ in range(len(Times))]
    frailty_model = Frailty(X, Times, Cens, frailty, no_frailty_model.betas)
    frailty_paths = []
    observable_paths = []
    for k in range(30):
        frailty_model.draw()
        frailty_model.fit(frailty=True)
        print(frailty_model.eta)
        print(frailty_model.betas)
        print(k)
        if k > 10:
            frailty_paths += [[frailty_model.eta[0] * frailty_model.Y[i] for i in range(len(Y))]]
            observable_paths += [np.mean([np.sum([frailty_model.betas*[p] * X[t][i][p]] for p in range(frailty_model.p)) for i in range(frailty_model.n)]) for t in range(frailty_model.T)]

    plt.figure()
    plt.plot([eta * Y[k] for k in range(len(Y))], color="red")
    plt.plot(np.array(frailty_paths).T, color="blue", alpha=0.05)
    plt.show()
