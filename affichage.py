# noinspection PyInterpreter
import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_generator import generate_default_hurlin, generate_default
from PLTR import pltr
from bayesian_classifier import bayesian_classifier
import time
import matplotlib.pyplot as plt
from matplotlib import colors


def get_Hurlin_graphs(alpha=0.1):
    """

    :param alpha:
    :return:
    """
    debut = time.time()
    logit_scores_nl = []
    logit_scores = []
    rf_scores = []
    ltr_scores = []
    pltr_scores = []
    params = [k for k in range(4, 20 + 2, 2)]
    n_sim = 10
    n_indiv = 500
    n_indiv_test = 250
    for param in params:
        print(param)
        temp_logit_scores = []
        temp_rf_scores = []
        temp_ltr_scores = []
        temp_logit_scores_nl = []
        temp_pltr_scores = []
        for _ in range(n_sim):
            Y, X, X_test, Y_test, betas, gammas, deltas = generate_default_hurlin(n_indiv=n_indiv,
                                                                                  n_indiv_test=n_indiv_test,
                                                                                  n_params=param,
                                                                                  non_linear=False,
                                                                                  q = param // 2)
            clf = LogisticRegression().fit(X, Y)
            temp_logit_scores += [clf.score(X_test, Y_test)]
            rf = RandomForestClassifier().fit(X, Y)
            temp_rf_scores += [rf.score(X_test, Y_test)]

            ltr = pltr(penalization=0)
            ltr.fit(X, Y, X_test)
            score_ltr = ltr.clf.score(ltr.X_test, Y_test)
            p_ltr = pltr(penalization=0.1)
            p_ltr.fit(X, Y, X_test)
            score_pltr = p_ltr.clf.score(p_ltr.X_test, Y_test)


            Y, X, X_test, Y_test, betas, gammas, deltas = generate_default_hurlin(n_indiv=n_indiv,
                                                                                  n_indiv_test=n_indiv_test,
                                                                                  n_params=param,
                                                                                  non_linear=True,
                                                                                  q = param//2)
            clf = LogisticRegression().fit(X, Y)
            temp_logit_scores_nl += [clf.score(X_test, Y_test)]



            temp_ltr_scores += [score_ltr]
            temp_pltr_scores += [score_pltr]

        logit_scores += [[np.sort(temp_logit_scores)[int(alpha * len(temp_logit_scores))],
                          np.mean(temp_logit_scores),
                          np.sort(temp_logit_scores)[int((1 - alpha) * len(temp_logit_scores))]]]

        rf_scores += [[np.sort(temp_rf_scores)[int(alpha * len(temp_rf_scores))],
                       np.mean(temp_rf_scores),
                       np.sort(temp_rf_scores)[int((1 - alpha) * len(temp_rf_scores))]]]

        logit_scores_nl += [[np.sort(temp_logit_scores_nl)[int(alpha * len(temp_logit_scores_nl))],
                             np.mean(temp_logit_scores_nl),
                             np.sort(temp_logit_scores_nl)[int((1 - alpha) * len(temp_logit_scores_nl))]]]

        ltr_scores += [[np.sort(temp_ltr_scores)[int(alpha * len(temp_ltr_scores))],
                        np.mean(temp_ltr_scores),
                        np.sort(temp_ltr_scores)[int((1 - alpha) * len(temp_ltr_scores))]]]

        pltr_scores += [[np.sort(temp_pltr_scores)[int(alpha * len(temp_pltr_scores))],
                        np.mean(temp_pltr_scores),
                       np.sort(temp_pltr_scores)[int((1 - alpha) * len(temp_pltr_scores))]]]

    fin = time.time()
    print("temps d'execution : ", fin - debut)
    return [logit_scores, rf_scores, logit_scores_nl, ltr_scores, pltr_scores], params, ["linear Logit", "Random Forest",
                                                                            "Non linear logit", "ltr", "pltr"]


def affichage(X, Y, label):
    """
    Permet d'afficher les graphiques dans un format adéquat.
    :param X:
    :param Y:
    :param label:
    :return:
    """

    plt.figure()
    n = len(Y)
    color = iter(plt.cm.rainbow(
        np.linspace(0, 1, n)))  # Permet d'obtenir une liste de couleurs sur laquelle on itere pour tracer le graphique.
    plt.ylabel("Proportion of good predictions")
    plt.xlabel("Number of predictors")
    for i in range(len(Y)):
        c = next(color)
        for j in range(len(Y[i][0])):
            if j == 1:
                plt.plot(X, [Y[i][k][j] for k in range(len(Y[i]))], label=label[i], c=c)
            #if j == 0:
            #    plt.plot(X, [Y[i][k][j] for k in range(len(Y[i]))], label=label[i] + " lower bond", alpha=0.2, c=c)
            #if j == 2:
            #    plt.plot(X, [Y[i][k][j] for k in range(len(Y[i]))], label=label[i] + " upper bond", alpha=0.2, c=c)
    plt.legend()
    plt.grid()
    plt.show()


def naive_bayes_classifieur(params=None, n_gen=10):
    if params is None:
        params = [k for k in range(2, 22, 2)]
    score_nb = []
    score_logit = []
    for n_param in params:
        print("############ {param} ##############".format(param=n_param))
        score_nb_temp = []
        score_logit_temp = []
        for _ in range(n_gen):
            Y, X, X_test, Y_test, betas = generate_default(n_indiv=5000, n_indiv_test=2500, n_params=n_param)
            clf = bayesian_classifier()
            clf.fit(X, Y)
            logit = sklearn.linear_model.LogisticRegression().fit(X, Y)
            score_nb_temp += [clf.score(X_test, Y_test)]
            score_logit_temp += [logit.score(X_test, Y_test)]
        score_nb += [np.mean(score_nb_temp)]
        score_logit += [np.mean(score_logit_temp)]
    return [score_nb, score_logit], params, ["Naive Bayesian classifier", "logistic regression"]


if __name__ == "__main__":
    score, params, label = get_Hurlin_graphs()
    affichage(params, score, label)
