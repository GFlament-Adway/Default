import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from data_generator import generate_default_hurlin, pltr

def get_Hurlin_graphs():
    logit_scores_nl = []
    logit_scores = []
    rf_scores = []
    ltr_scores = []
    params = [k for k in range(4, 20 + 2, 2)]
    n_sim = 1
    n_indiv = 5000
    n_indiv_test = 2500
    for param in params:
        temp_logit_scores = []
        temp_rf_scores = []
        temp_ltr_scores = []
        temp_logit_scores_nl = []
        for _ in range(n_sim):
            Y, X, X_test, Y_test, betas, gammas = generate_default_hurlin(n_indiv=n_indiv, n_indiv_test=n_indiv_test, n_params=param,
                                                                          non_linear=False)
            clf = LogisticRegression().fit(X, Y)
            temp_logit_scores += [clf.score(X_test, Y_test)]
            rf = RandomForestClassifier().fit(X, Y)
            temp_rf_scores += [rf.score(X_test, Y_test)]
            Y, X, X_test, Y_test, betas, gammas = generate_default_hurlin(n_indiv=n_indiv, n_indiv_test=n_indiv_test, n_params=param,
                                                                          non_linear=True)
            clf = LogisticRegression().fit(X, Y)
            ltr, score = pltr(X,Y, X_test, Y_test)
            temp_logit_scores_nl += [clf.score(X_test, Y_test)]
            temp_ltr_scores += [score]

        logit_scores += [np.mean(temp_logit_scores)]
        rf_scores += [np.mean(temp_rf_scores)]
        logit_scores_nl += [np.mean(temp_logit_scores_nl)]
        ltr_scores += [np.mean(temp_ltr_scores)]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.ylabel("Proportion of good predictions")
    plt.xlabel("Number of predictors")
    plt.plot(params, logit_scores, label='Logistic score')
    plt.plot(params, logit_scores_nl, label="Non linear logistic score")
    plt.plot(params, rf_scores, label="Random Forest score")
    plt.plot(params, ltr_scores, label="ltr score")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    Y, X, X_test, Y_test, deltas, gammas = generate_default_hurlin(n_params=10, non_linear=False)
    get_Hurlin_graphs()

