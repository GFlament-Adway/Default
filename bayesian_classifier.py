from data_generator import generate_default_hurlin, generate_default
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import norm


class bayesian_classifier():
    """

    :param data:
    :return:
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        n_indiv = np.array(X).shape[0]
        n_param = np.array(X).shape[1]
        classes = np.unique(Y)
        self.classes = classes

        X_by_classes = [[[X[i][p] for i in range(n_indiv) if Y[i] == c] for p in range(n_param)] for c in classes]
        self.means = {c: [np.mean(X_by_classes[c][p]) for p in range(n_param)] for c in classes}
        self.vars = {c: [np.var(X_by_classes[c][p]) for p in range(n_param)] for c in classes}

    def predict(self, X):
        n_params = len(X)
        posterior = {c: np.prod([norm.pdf(X[p], loc=self.means[c][p], scale=self.vars[c][p]) for p in range(n_params)])
                     for c in self.classes}
        return list(posterior.keys())[np.argmax(list(posterior.values()))]

    def score(self, X, Y):
        return 1 - np.sum([abs(clf.predict(X[i]) - Y[i]) for i in range(len(X))]) / (len(X))


if __name__ == "__main__":
    Y, X, X_test, Y_test, betas = generate_default(n_indiv=500, n_indiv_test=250, n_params=2)
    clf = bayesian_classifier()
    clf.fit(X, Y)
    print("Guillaume : ", clf.score(X_test, Y_test), clf.score(X, Y))
    clf_sklearn = GaussianNB().fit(X, Y)
    print("sklearn : ", clf_sklearn.score(X_test, Y_test), clf.score(X, Y))
