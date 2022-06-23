from utils import load_params, output_latex, save_figure
from sampler import Frailty
import numpy as np
from data_generator import get_data

if __name__ == "__main__":

    params = load_params()

    np.random.seed(int(params["seed"]))

    N = params["samples"]
    X, Y, Times, Cens, betas, eta = get_data(params["n_obs"], params["max_time"], kwargs=params)
    print("Real parameters : ", betas, eta)
    print("Censorship rate", np.sum(Cens) / len(Times))
    frailty = [[0 for _ in range(len(Y))] for _ in range(N)]
    # First value of the frailty
    print("Mean evt time : ", np.mean(Times))
    no_frailty_model = Frailty(X, Times, Cens, frailty, betas=params["init"]["betas"], eta=0)
    print("Initial values :", no_frailty_model.betas)
    print("#################### First step of Duffie ################")
    print("############ Estimating betas without frailty ############""")

    no_frailty_model.fit(frailty=False)
    print(no_frailty_model.betas)

    print("############ Second step, estimating eta and frailty ##################")

    frailty = [[0 for k in range(len(Y))] for _ in range(N)]
    # Start without frailty
    print(np.array(frailty).shape)
    frailty_model = Frailty(X, Times, Cens, frailty, betas=no_frailty_model.betas, eta=params["init"]["eta"])
    print("Betas ", frailty_model.betas)
    frailty_paths = []
    observable_paths = []
    likes = []
    for k in range(params["n_iter"]):
        print(k)
        frailty_model.fit(frailty=True)
        print("Log like : ", frailty_model.log_likelihood)
        print("Estimated : ", frailty_model.betas, frailty_model.eta)
        print("Real : ", betas, eta)
        frailty_model.draw()
        likes += [frailty_model.log_likelihood]
        frailty_paths += [[frailty_model.eta[0] * frailty_model.Y[a][i] for i in range(len(Y))] for a in range(N)]
        # observable_paths += [np.mean([np.sum([frailty_model.betas*[p] * X[t][i][p]] for p in range(frailty_model.p)) for i in range(frailty_model.n)]) for t in range(frailty_model.T)]


    if params["output"] == "latex":
        output_latex(params)

    if params["save_figures"] == "True":
        save_figure(frailty_paths, )
