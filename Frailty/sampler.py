import numpy as np
import optimizer


class frailty():
    def __init__(self, X, Times, Cens):
        self.last_draw = 0
        self.X = X
        self.Times = Times
        self.T = len(X)
        self.p = len(X[0][0])
        self.n = len(X[0])
        self.C = Cens
        self.betas = [0 for _ in range(self.p)]

    def draw(self):
        self.last_draw = np.random.normal(mean=self.last_draw, scale=2)
        return self.last_draw

    def likelihood(self, betas):
        """
        See equation 6.4 from D. Duffie
        :param Y:
        :return:
        """
        self.betas = betas
        intensities = [[np.exp(-np.sum([betas[j] * self.X[k][i][j] for j in range(self.p)])) for k in range(self.T)] for
                       i in range(self.n)]
        likelihood = 1
        for i in range(self.n):
            likelihood *= (np.exp(
                -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                            float(self.Times[i]) - int(self.Times[i]))) ** (1 - self.C[i])
                           * ((intensities[i][int(self.Times[i])]) * np.exp(
                        -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                                    self.Times[i] - int(self.Times[i])))) ** self.C[i])
        return likelihood

    def fit(self):
        gen_algo = optimizer.gen_algo(pop_size=100, n_params=self.p, func=self.likelihood, alpha=0.5, mutate_rate=0.05)
        for _ in range(10):
            gen_algo.evaluate_fitness()
            gen_algo.generate_next_gen()
            gen_algo.evaluate_fitness()
            print(gen_algo.current_pop[gen_algo.current_fitness.index(np.max(gen_algo.current_fitness))])
            print(np.max(gen_algo.current_fitness))

    def pred(self, X_nex):
        intensities = [[np.exp(-np.sum([betas[j] * X_nex[k][i][j] for j in range(self.p)])) for k in range(self.T)] for
                       i in range(self.n)]
        t = np.arange(self.T)
        L = np.array([[np.sum(intensities[k][:i]) for k in range(self.n)] for i in t]).T
        generated_times = []
        for i in range(self.n):
            U = np.random.uniform(0, 1)
            value = [x for x in L[i] if x <= -np.log(1 - U)][-1]
            idx = np.where(L[i] == value)[0][0]
            t = ((-np.log(1 - U) - value) + intensities[i][idx] * idx) / intensities[i][idx]
            generated_times += [t]
        return generated_times


if __name__ == "__main__":
    #np.random.seed(12)
    from data_generator import get_data

    X, Y, Times, Cens, betas = get_data(100, 10, 3, censure_rate=0.2)
    print("Parameters to estimate : ", betas)
    model = frailty(X, Times, Cens)
    print("Likelihood of the true parameters : ", model.likelihood(betas=betas))
    model.fit()
