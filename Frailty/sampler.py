import numpy as np
from scipy.optimize import minimize
from utils import load_params, output_latex, save_figure

class Frailty():
    def __init__(self, X, Times, Cens, Y, N=None, betas=None, eta=None, optim=None):
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
        self.Y = Y
        self.frailty = [[eta * y for y in Y[a]] for a in range(len(Y))]

        if optim is None :
            self.optim = "full_like"
        else:
            self.optim = optim
        if N is None:
            self.N = np.array(Y).shape[0]
        else:
            self.N = N
        self.betas = betas
        self.eta = eta

    def likelihood(self, param, *args):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        :return:
        """
        like = []
        likelihood = []
        if np.all(np.array(self.Y) == 0) or args[2] is False:
            """
            Case during the first step of Duffie, no need to compute all Frailty path as they are all equal.
            """
            if args[0] == "beta":
                intensities = [
                    [max(min(np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * args[1][0][k]), 10e20), 10e-10) for
                     k in
                     range(int(self.Times[i]) + 1)] for
                    i in range(self.n)]

            elif args[0] == "eta":
                intensities = [
                    [max(min(np.exp(np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param * args[1][0][k]), 10e20), 10e-10)
                     for k in
                     range(int(self.Times[i]) + 1)] for
                    i in range(self.n)]
            elif args[0] == "Y":
                intensities = [
                    [max(min(np.exp(
                        np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * param[k]),
                        10e20), 10e-10)
                        for k in
                        range(int(self.Times[i]) + 1)] for
                    i in range(self.n)]

            for i in range(self.n):
                int_intensity = -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                        float(self.Times[i]) - int(self.Times[i]))
                assert intensities[i][int(self.Times[i])] > 0, "{inten}".format(inten=intensities[i])
                like += [(1 - self.C[i]) * (int_intensity + np.log(intensities[i][int(self.Times[i])])) + self.C[i] * int_intensity]
            return -np.sum(like)

        else:
            for a in range(self.N):
                #print(a)
                log_likelihood = []
                if args[0] == "beta":
                    intensities = [
                        [max(min(np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * args[1][a][k]), 10e20), 10e-10)
                         for k in
                         range(int(self.Times[i]) + 1)] for
                        i in range(self.n)]

                elif args[0] == "eta":
                    intensities = [
                        [max(min(np.exp(
                            np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param * args[1][a][k]), 10e20), 10e-10)
                         for k in
                         range(int(self.Times[i]) + 1)] for
                        i in range(self.n)]
                elif args[0] == "Y":
                    intensities = [
                        [max(min(np.exp(
                            np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * param[k]),
                            10e20), 10e-10)
                            for k in
                            range(int(self.Times[i]) + 1)] for
                        i in range(self.n)]
                like = []
                for i in range(self.n):
                    int_intensity = -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                            float(self.Times[i]) - int(self.Times[i]))
                    assert intensities[i][int(self.Times[i])] > 0, "{inten}".format(inten=intensities[i])
                    like += [(1 - self.C[i]) * (int_intensity + np.log(intensities[i][int(self.Times[i])])) + self.C[i] * int_intensity]

                log_likelihood += [np.sum(like)]
            return -np.mean(log_likelihood)

    def fit(self, method=None, frailty=True):
        if frailty:
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y, True), method=method,
                                       options={"maxiter": 5})
            self.betas = self.parameters["x"]
            if self.optim == "full_like":
                self.parameters = minimize(self.likelihood, self.Y[0], args=("Y", self.Y[0], True), method=method,
                                           options={"maxiter": 3})
                self.Y = [self.parameters["x"] for _ in range(self.N)]


                self.parameters = minimize(self.likelihood, self.eta, args=("eta", self.Y, True), method=method,
                                           options={"maxiter": 3})
                self.eta = self.parameters["x"]
                self.frailty = [self.eta * y for y in self.Y]
            else:
                self.parameters = minimize(self.likelihood, self.eta, args=("eta", self.Y, True), method=method,
                                           options={"maxiter": 3})
                self.eta = self.parameters["x"]

            #self.eta = [1]
        else:
            self.parameters = minimize(self.likelihood, self.betas, args=("beta", self.Y, False), method=method,
                                       options={"maxiter": 5})
            self.betas = self.parameters["x"]
        self.log_likelihood = -self.parameters["fun"]

    def draw(self):
        """
        Draw according to appendix D of D. Duffie measuring corporate default risk.
        :return:
        """
        #print("Draw")
        acceptance_rate = []
        for a in range(self.N):
            for k in range(self.T):
                y_k = np.random.normal(self.Y[a][k], 2)
                #y_k = self.frailty[a][k]
                new_frailty = [[y if i != k else y_k for i, y in enumerate(self.Y[a])]]
                new_like = self.likelihood(self.eta[0], "eta", new_frailty, False)
                old_like = self.likelihood(self.eta[0], "eta", [self.Y[a]], False)
                #print(np.exp(-new_like), np.exp(-old_like))
                U = np.random.uniform(0, 1)
                acceptance = min(np.exp(-new_like) / np.exp(-old_like), 1)

                if U < acceptance:
                    acceptance_rate += [1]
                    self.Y[a][k] = y_k
                else:
                    acceptance_rate += [0]
        print("mean acceptance rate : ", np.mean(acceptance_rate))