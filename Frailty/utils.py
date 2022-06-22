import json


def load_params(path="settings/settings.json"):
    """

    :param path: path to file that contains all the settings.
    :return: Dictionnary with all necessary settings.
    """
    with open(path) as json_file:
        data = json.load(json_file)

    return data


def output_latex(param):
    return """
    \\begin{tabular}{c|c}
    parameter & value \\\\
     Starting value of $\\beta $ """ + " & {beta}".format(beta=param["init"]["betas"]) + """
     \\\\
     \\hline
     Starting value of $\\eta$ &""" + " {eta} ".format(eta=param["init"]["eta"]) + """\\\\
     \\hline
     Real values of $\\beta$ &""" + " {beta} ".format(beta=param["real values"]["betas"]) + """\\\\
     \\hline
     Real value of $\\eta$ &""" + " {eta} ".format(eta=param["real values"]["eta"]) + """\\\\
     \\hline
     $n$ &""" + " {n} ".format(n=param["n_obs"]) + """\\\\
     \\hline
     $X$ &""" + " $\mathcal{U}"+ "({m}, {M})$ ".format(m=param["min_values_X"], M=param["max_values_X"]) + """\\\\
     \\hline
     $Censure$ &""" + " $\\exp"+ "({m})$ ".format(m=param["Censure"]) + """\\\\
     \\hline
     $Seed$ &""" "{s}$ ".format(s=param["seed"]) + """\\\\
     \\hline
\end{tabular}"""


def save_figure():
    pass

if __name__ == "__main__":
    print(output_latex(load_params()))
