import json
import numpy as np
import os
import matplotlib.pyplot as plt


def mse(x, y):
    assert len(x) == len(y)
    return np.mean([(float(x[k]) - float(y[k])) ** 2 for k in range(len(x))])


def check_setup(params):
    """
    Check files and directory setup before runing
    :return:
    """
    cwd = os.getcwd()
    existing_runs = len(os.listdir(cwd + "/run"))
    for k in range(params["n_run"]):
        if not os.path.isdir(cwd + "/run/run_{k}".format(k=k + existing_runs)):
            print(cwd + "/run/run_{k}".format(k=k + existing_runs))
            os.mkdir(cwd + "/run/run_{k}".format(k=k + existing_runs))


def load_params(path="settings/settings.json"):
    """

    :param path: path to file that contains all the settings.
    :return: Dictionnary with all necessary settings.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data


def output_latex(param, results, run, existing_runs):
    result_tab = """ \\begin{tabular}{c|c|c|c|c}
    Iteration & beta & eta & MSE Frailty & Frailty mean\\\\ 
    \\hline
    """
    for k in results.keys():
        result_tab += "{k} & {beta} & {eta} & {mse} & {frailty_mean}\\\\ \hline".format(k=k, beta=results[k]["estimated betas"],
                                                                        eta=results[k]["estimated eta"],
                                                                        mse=results[k]["mse"],
                                                                                        frailty_mean=results[k]["frailty mean"])

    result_tab += """
    \\end{tabular}
    \\end{center}
    """
    full_string = """
    \\begin{center}
    \\begin{tabular}{c|c}
    parameter & value \\\\
     Starting value of $\\beta $ """ + " & {beta}".format(beta=param["init"]["betas"]) + """ \\\\
     \\hline
     Starting value of $\\eta$ &""" + " {eta} ".format(eta=param["init"]["eta"]) + """\\\\ 
     \\hline
     Real values of $\\beta$ &""" + " {beta} ".format(beta=param["real values"]["betas"]) + """\\\\
     \\hline
     Real value of $\\eta$ &""" + " {eta} ".format(eta=param["real values"]["eta"]) + """ \\\\
     \\hline
     $n$ &""" + " {n} ".format(n=param["n_obs"]) + """ \\\\
     \\hline
     $X$ &""" + " $\mathcal{U}" + "({m}, {M})$ ".format(m=param["min_values_X"], M=param["max_values_X"]) + """ \\\\
     \\hline
     Censure &""" + " $\\exp" + "({m})$ ".format(m=param["Censure"]) + """ \\\\
     \\hline
     Seed &""" "${s}$ ".format(s=param["seed"]) + """ \\\\
     \\hline
\end{tabular}""" + result_tab
    with open("run/run_{k}/result_summary.txt".format(k=run + existing_runs), "w") as txt_file:
        txt_file.write(full_string)

    if param["savefig"] == "True":
        plt.figure()
        plt.plot(results[run]["Y"], color="red", label="True value")
        plt.plot(np.array(results[run]["frailty path"]).T, color="blue", alpha=0.1)
        plt.legend()
        plt.savefig("run/run_{k}/frailty_path_{l}".format(k=run+existing_runs, l=run+existing_runs))

        mat = np.matmul(results[run]["X"], results[run]["estimated betas"]).T
        mat += results[run]["estimated eta"] * np.mean(results[run]["frailty path"][0])
        hat_intensities = np.clip(np.exp(mat), 10e-120, 10e120)

        mat = np.matmul(results[run]["X"], results[run]["betas"]).T
        mat += results[run]["eta"] * np.mean(results[run]["Y"][0])
        intensities = np.clip(np.exp(mat), 10e-120, 10e120)

        for a in range(len(hat_intensities)):
            plt.figure()
            plt.plot(hat_intensities[a], label="hat")
            plt.plot(intensities[a], label="true")
            plt.legend()
            plt.savefig("run/run_{k}/hazard_rate_indiv_{a}".format(k=run+existing_runs, a=a))

    return full_string



if __name__ == "__main__":
    print(output_latex(load_params(),
                       {1: {"estimated betas": [1, 2, 1, 2, 1], "estimated eta": 0.1, "mse": [12, 20, 10]}}))
