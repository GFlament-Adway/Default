import json
import numpy as np
import os
import matplotlib.pyplot as plt


def mse(x,y):
    assert len(x) == len(y)
    return np.mean([(x[k] - y[k])**2 for k in range(len(x))])


def check_setup(params):
    """
    Check files and directory setup before runing
    :return:
    """
    cwd = os.getcwd()
    for k in range(params["n_run"]):
        if not os.path.isdir(cwd + "/run_{k}".format(k=k)):
            print(cwd + "/run_{k}".format(k=k))
            os.mkdir(cwd + "/run_{k}".format(k=k))



def load_params(path="settings/settings.json"):
    """

    :param path: path to file that contains all the settings.
    :return: Dictionnary with all necessary settings.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data


def output_latex(param, results, run):
    result_tab = """ \\begin{tabular}{c|c|c|c}
    Iteration & beta & eta & MSE Frailty\\\\ 
    \\hline
    """
    for k in results.keys():
        result_tab += "{k} & {beta} & {eta} & {mse} \\\\ \hline".format(k=k, beta = results[k]["estimated betas"], eta =results[k]["estimated eta"], mse = results[k]["mse"])

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
     Starting value of $\\eta$ &""" + " {eta} ".format(eta = param["init"]["eta"]) + """\\\\ 
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
\end{tabular}"""+ result_tab
    with open("run_{k}/file.txt".format(k=run), "w") as txt_file:
        txt_file.write(full_string)

    if param["savefig"] == "True":
        plt.figure()
        plt.plot(np.array(results["frailty path"][-1]).T, alpha=0.1)
        plt.plot(results["Y"])
        plt.savefig("run_{k}/frailty_path".format(k=run))

    return full_string




def save_figure():
    pass


if __name__ == "__main__":
    print(output_latex(load_params(), {1 : {"estimated betas" : [1,2,1,2,1], "estimated eta": 0.1, "mse": [12,20,10]}}))
