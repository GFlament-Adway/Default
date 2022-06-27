from utils import load_params, output_latex, mse, check_setup
from sampler import Frailty
import numpy as np
from data_generator import get_data

if __name__ == "__main__":

    params = load_params()
    n_run = len(params["run"])
    params.update({"n_run": n_run})
    check_setup(params)
    for i in range(params["n_run"]):
        np.random.seed(int(params["run"][i]["seed"]))

        N = params["run"][i]["samples"]
        X, Y, Times, Cens, betas, eta = get_data(params["run"][i]["n_obs"], params["run"][i]["max_time"],
                                                 kwargs=params["run"][i])
        print("Real parameters : ", betas, eta)
        print("Censorship rate", np.sum(Cens) / len(Times))
        Y_hat = [[0 for _ in range(len(Y))] for _ in range(N)]
        # First value of the frailty
        print("Mean evt time : ", np.mean(Times))
        no_frailty_model = Frailty(X, Times, Cens, Y_hat, betas=params["run"][i]["init"]["betas"], eta=0)
        print("Initial values :", no_frailty_model.betas)
        print("#################### First step of Duffie ################")
        print("############ Estimating betas without frailty ############""")

        no_frailty_model.fit(frailty=False)
        print(no_frailty_model.betas)

        print("############ Second step, estimating eta and frailty ##################")
        if params["run"][i]["init"]["Y"] == "0":
            Y_hat = [[0 for k in range(len(Y))] for _ in range(N)]
        elif params["run"][i]["init"]["Y"] == "random":
            Y_hat = [
                [Y[k] + np.random.normal(0, params["run"][i]["init"]["Y_var"]) for k
                 in range(len(Y))] for _ in
                range(N)]
        # Start without frailty
        frailty_model = Frailty(X, Times, Cens, Y_hat, betas=no_frailty_model.betas,
                                eta=params["run"][i]["init"]["eta"], optim=params["run"][i]["optimization_method"])

        print(np.array(Y).shape)
        print("Init mse : ",
              [mse(frailty_model.frailty[a], [params["run"][i]["real values"]["eta"] * Y[t] for t in range(len(Y))]) for a in
               range(N)])
        print("Betas ", frailty_model.betas)
        frailty_paths = []
        observable_paths = []
        likes = []
        results = {}
        for k in range(params["run"][i]["n_iter"]):
            print(k)
            frailty_model.fit(frailty=True)
            print("- log like : ", frailty_model.log_likelihood)
            print("Estimated : ", frailty_model.betas, frailty_model.eta)
            print("Real : ", betas, eta)
            likes += [frailty_model.log_likelihood]
            if params["run"][i]["optimization_method"] == "gibbs":
                frailty_model.draw()
                frailty_paths = [[frailty_model.eta[0] * frailty_model.Y[a][i] for i in range(len(Y))] for a in
                                 range(N)]
                frailty_path_error = [
                    mse(frailty_paths[a], [params["run"][i]["init"]["eta"] * Y[t] for t in range(len(Y))])
                    for a in range(N)]
            if params["run"][i]["optimization_method"] == "full_like":
                frailty_path_error = mse([frailty_model.eta[0] * frailty_model.Y[0][i] for i in range(len(Y))],
                                         [params["run"][i]["init"]["eta"] * y for y in Y])
                frailty_paths = [[frailty_model.eta[0] * frailty_model.Y[0][i] for i in range(len(Y))] for _ in
                                 range(N)]


            results.update({k: {"estimated betas": np.round(frailty_model.betas, 3),
                                "estimated eta": np.round(frailty_model.eta, 3),
                                "mse": np.round(np.mean(frailty_path_error), 3), "frailty path": frailty_paths,
                                "Y": Y}})
            print("frailty mse : ", results[k]["mse"])

    if params["run"][i]["output"] == "latex":
        output_latex(params["run"][i], results, run=i)
