import numpy as np
import sklearn.linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from data_generator import generate_default_hurlin, generate_default
from PLTR import pltr
from bayesian_classifier import bayesian_classifier
import time

def get_Hurlin_graphs():
    debut = time.time()
    logit_scores_nl = []
    logit_scores = []
    rf_scores = []
    ltr_scores = []
    params = [k for k in range(4, 20 + 2, 2)]
    n_sim = 10
    n_indiv = 5000
    n_indiv_test = 2500
    for param in params:
        print(param)
        temp_logit_scores = []
        temp_rf_scores = []
        temp_ltr_scores = []
        temp_logit_scores_nl = []
        for _ in range(n_sim):
            Y, X, X_test, Y_test, betas, gammas, deltas = generate_default_hurlin(n_indiv=n_indiv, n_indiv_test=n_indiv_test, n_params=param,
                                                                          non_linear=False)
            clf = LogisticRegression().fit(X, Y)
            temp_logit_scores += [clf.score(X_test, Y_test)]
            ltr = pltr()
            ltr.fit(X, Y, X_test)
            score = ltr.clf.score(ltr.X_test, Y_test)
            rf = RandomForestClassifier().fit(X, Y)
            temp_rf_scores += [rf.score(X_test, Y_test)]

            Y, X, X_test, Y_test, betas, gammas, deltas = generate_default_hurlin(n_indiv=n_indiv, n_indiv_test=n_indiv_test, n_params=param,
                                                                          non_linear=True)
            clf = LogisticRegression().fit(X, Y)
            temp_logit_scores_nl += [clf.score(X_test, Y_test)]
            temp_ltr_scores += [score]

        logit_scores += [np.mean(temp_logit_scores)]
        rf_scores += [np.mean(temp_rf_scores)]
        logit_scores_nl += [np.mean(temp_logit_scores_nl)]
        ltr_scores += [np.mean(temp_ltr_scores)]
    fin = time.time()
    print("temps d'exec : ", fin - debut)
    return [logit_scores, rf_scores, logit_scores_nl, ltr_scores], params

def affichage(X, Y, label):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.ylabel("Proportion of good predictions")
    plt.xlabel("Number of predictors")
    for i in range(len(Y)):
        plt.plot(X, Y[i], label=label[i])
    plt.legend()
    plt.grid()
    plt.show()

def naive_bayes_classifieur(params =  None, n_gen = 10):
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
            logit = sklearn.linear_model.LogisticRegression().fit(X,Y)
            score_nb_temp += [clf.score(X_test, Y_test)]
            score_logit_temp += [logit.score(X_test, Y_test)]
        score_nb += [np.mean(score_nb_temp)]
        score_logit += [np.mean(score_logit_temp)]
    return [score_nb, score_logit], params, ["Naive Bayesian classifier", "logistic regression"]

if __name__ == "__main__":
    score, params = naive_bayes_classifieur()
    affichage(params, score, ["linear Logit", "Random Forest", "Non linear logit", "ltr"])