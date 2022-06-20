import numpy as np
from scipy.optimize import minimize


class Frailty():
    def __init__(self, X, Times, Cens, frailty, N=None, betas=None):
        """

        :param X: Covariates
        :param Times: Event times
        :param Cens: Censure times
        :param frailty: frailty paths
        :param N: number of frailty to approximate the expectation 6.5 of D. Duffie Measuring corporate default risk.
        :param betas: parameters to be estimated
        """

        self.last_draw = 0
        self.X = X
        self.Times = Times
        self.T = len(X)
        self.p = len(X[0][0])
        self.n = len(X[0])
        self.C = Cens
        self.Y = frailty
        if N is None:
            self.N = np.array(frailty).shape[0]
        else:
            assert np.array(frailty).shape[0] == N
            self.N = N
        if betas is None:
            """
            If beta is not given, start at the real parameter 
            @todo : make sure the vector is of correct length.
            """
            self.betas = [-1, -1.2, -0.65, -0.25, 1.55]
        else:
            self.betas = betas
        self.eta = 0.12

    def draw(self):
        """

        :return:
        """
        self.last_draw = np.random.normal(mean=self.last_draw, scale=0.5)
        return self.last_draw

    def likelihood(self, param, *args):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        :return:
        """
        log_likelihood = []
        if np.all(np.array(frailty) == 0):
            """
            Case during the first step of Duffie, no need to compute all Frailty path as they are all equal.
            """
            if args[0] == "beta":
                intensities = [
                    [np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * args[1][k]) for k in
                     range(int(self.Times[i]) + 1)] for
                    i in range(self.n)]

            elif args[0] == "eta":
                intensities = [
                    [np.exp(np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param * args[1][k]) for k in
                     range(int(self.Times[i]) + 1)] for
                    i in range(self.n)]

            for i in range(self.n):
                log_likelihood += [(1 - self.C[i]) * (
                        -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                        float(self.Times[i]) - int(self.Times[i]))) + \
                                  self.C[i] * (np.log(intensities[i][int(self.Times[i])]) - np.sum(
                    intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                                                       self.Times[i] - int(self.Times[i])))]
            return -log_likelihood[0]

        else:
            for _ in range(self.N):
                if args[0] == "beta":
                    intensities = [
                        [np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * args[1][k]) for k in
                         range(int(self.Times[i]) + 1)] for
                        i in range(self.n)]

                elif args[0] == "eta":
                    intensities = [
                        [np.exp(np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param * args[1][k]) for k in
                         range(int(self.Times[i]) + 1)] for
                        i in range(self.n)]

                for i in range(self.n):
                    log_likelihood += [(1 - self.C[i]) * (
                            -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                            float(self.Times[i]) - int(self.Times[i]))) + \
                                      self.C[i] * (np.log(intensities[i][int(self.Times[i])]) - np.sum(
                        intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                                                           self.Times[i] - int(self.Times[i])))]
            return -np.mean(log_likelihood)

    def fit(self, method=None, frailty=True):
        if frailty:
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y), method=method)
            self.betas = self.parameters["x"]
            self.parameters = minimize(self.likelihood, self.eta, args=("eta", self.Y), method=method)
            self.eta = self.parameters["x"]
        else:
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y), method=method)
            self.betas = self.parameters["x"]
        self.log_likelihood = -self.parameters["fun"]

    def draw(self):
        """
        Draw according to appendix D of D. Duffie measuring corporate default risk.
        :return:
        """
        for k in range(self.T):
            y_k = np.random.normal(self.Y[k], 2)
            new_frailty = [y if i != k else y_k for i, y in enumerate(self.Y)]
            new_like = self.likelihood(self.eta, "eta", new_frailty)
            old_like = self.likelihood(self.eta, "eta", self.Y)
            U = np.random.uniform(0,1)
            acceptance = min(np.exp(-new_like) / np.exp(-old_like), 1)
            if U < acceptance:
                self.Y[k] = y_k


if __name__ == "__main__":
    np.random.seed(1234)
    from data_generator import get_data
    import matplotlib.pyplot as plt

    N = 20
    X, Y, Times, Cens, betas, eta = get_data(100, 30, censure_rate=0)
    print("Real parameters : ", betas, eta)
    print("Censorship rate", np.sum(Cens)/len(Times))
    frailty = [0 for _ in range(len(Y)) for _ in range(N)]
    print("Frailty mean ", np.mean(Y))
    print("Mean evt time : ", np.mean(Times))
    no_frailty_model = Frailty(X, Times, Cens, frailty)
    print("#################### First step of Duffie ################")
    print("############ Estimating betas without frailty ############""")
    no_frailty_model.fit(frailty=False)
    print(no_frailty_model.betas)
    print("############ Second step, estimating eta and frailty ##################")
    frailty = [np.random.normal(0, 1) for _ in range(len(Times))]
    frailty_model = Frailty(X, Times, Cens, frailty, None)
    print("Betas ", frailty_model.betas)
    frailty_paths = []
    observable_paths = []
    likes = []
    for k in range(200):
        frailty_model.draw()
        frailty_model.fit(frailty=True)
        print("Log like : ", frailty_model.log_likelihood)
        print("Estimated : ", frailty_model.betas, frailty_model.eta)
        print("Real : ", betas, eta)
        likes += [frailty_model.log_likelihood]
        if k > 100:
            frailty_paths += [[frailty_model.eta[0] * frailty_model.Y[i] for i in range(len(Y))]]
            #observable_paths += [np.mean([np.sum([frailty_model.betas*[p] * X[t][i][p]] for p in range(frailty_model.p)) for i in range(frailty_model.n)]) for t in range(frailty_model.T)]

    plt.figure()
    plt.plot(likes)
    plt.draw()

    plt.figure()
    plt.plot(np.array(frailty_paths).T, color="blue", alpha=0.1)
    plt.plot([Y[k]*eta for k in range(len(Y))])
    plt.show()
