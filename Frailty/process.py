import numpy as np

class Brownian_motion():
    def __init__(self):
        pass

    def get_dw(self, T):
        return np.random.normal(0, 1, T)

    def get_W(self, T):
        dw = self.get_dw(T)
        dw_cs = np.cumsum(dw)
        return np.insert(dw_cs, 0, 0)[:-1]


class OU_process():
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_OU(self, T):
        t = np.arange(T)
        exp_a_t = np.exp(-self.alpha * t)
        bm = Brownian_motion()
        dw = bm.get_dw(T)
        return self.gamma * exp_a_t + self.gamma * (1 - exp_a_t) + self.beta * exp_a_t * np.cumsum(np.exp(self.alpha * t) * dw)

if __name__ == "__main__":
    alpha = 0.05
    beta = 0.1
    gamma = 0
    T = 1000

    BM = Brownian_motion()
    BM_process = BM.get_W(T)
    OU = OU_process(alpha, beta, gamma)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(OU.get_OU(T))
    plt.show()