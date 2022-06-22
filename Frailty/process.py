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
    def __init__(self, kappa, burn = 0):
        self.kappa = kappa
        self.burn = burn

    def get_OU(self, T, burn = None, var = 2):
        self.Y = [0]
        if burn is not None:
            self.burn = burn
        t = np.arange(self.burn + T)
        for _ in t:
            self.Y += [self.Y[-1]*(1-self.kappa) + np.random.normal(0, var)]
        return self.Y[self.burn:]
 
if __name__ == "__main__":
    kappa = 0.0018
    T = 3000

    BM = Brownian_motion()
    BM_process = BM.get_W(T)
    OU = OU_process(kappa, burn=100)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(OU.get_OU(T))
    plt.show()