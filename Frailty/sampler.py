import numpy as np
from scipy.optimize import minimize
from utils import load_params

class Frailty():
    def __init__(self, X, Times, Cens, frailty, N=None, betas=None, eta=None):
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
            self.N = N
        self.betas = betas
        self.eta = eta

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
        like = []
        likelihood = []
        if np.all(np.array(frailty) == 0) or args[2] is False:
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
            self.parameters = minimize(self.likelihood, self.eta, args=("eta", self.Y, True), method=method,
                                       options={"maxiter": 5})
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
                #y_k = self.Y[a][k]
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
if __name__ == "__main__":
    np.random.seed(1234)
    from data_generator import get_data
    import matplotlib.pyplot as plt

    params = load_params()
    N = params["samples"]
    X, Y, Times, Cens, betas, eta = get_data(params["n_obs"], params["max_time"])
    print("Real parameters : ", betas, eta)
    print("Censorship rate", np.sum(Cens) / len(Times))
    frailty = [[0 for _ in range(len(Y))] for _ in range(N)]
    print("Frailty mean ", np.mean(Y))
    print("Mean evt time : ", np.mean(Times))
    no_frailty_model = Frailty(X, Times, Cens, frailty, betas = params["init"]["betas"], eta = 0)
    print("Initial values :", no_frailty_model.betas)
    print("#################### First step of Duffie ################")
    print("############ Estimating betas without frailty ############""")
    no_frailty_model.fit(frailty=False)
    print(no_frailty_model.betas)

    print("############ Second step, estimating eta and frailty ##################")
    frailty = [[0 for k in range(len(Y))] for _ in range(N)]
    #Start without frailty
    print(np.array(frailty).shape)
    frailty_model = Frailty(X, Times, Cens, frailty, betas = no_frailty_model.betas, eta = params["init"]["eta"])
    print("Betas ", frailty_model.betas)
    frailty_paths = []
    observable_paths = []
    likes = []
    for k in range(20):
        print(k)
        frailty_model.fit(frailty=True)
        print("Log like : ", frailty_model.log_likelihood)
        print("Estimated : ", frailty_model.betas, frailty_model.eta)
        print("Real : ", betas, eta)
        frailty_model.draw()
        likes += [frailty_model.log_likelihood]
        frailty_paths += [[frailty_model.eta[0] * frailty_model.Y[a][i] for i in range(len(Y))] for a in range(N)]
        # observable_paths += [np.mean([np.sum([frailty_model.betas*[p] * X[t][i][p]] for p in range(frailty_model.p)) for i in range(frailty_model.n)]) for t in range(frailty_model.T)]

    print(frailty_paths)
    plt.figure()
    paths = np.array(frailty_paths).T
    plt.plot(paths, color="blue", alpha=0.6)
    plt.plot([Y[k] * eta for k in range(len(Y))], color="red")
    plt.show()
