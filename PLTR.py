import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from data_generator import generate_default_hurlin
from sklearn.linear_model import LogisticRegression

def tree_to_code(X, tree, feature_names):
    """
    :param X:
    :param tree:
    :param feature_names:
    :return:
    """
    tree_ = tree.tree_
    X_c = deepcopy(X)
    for i in range(len(X)):
        threshold = tree_.threshold[0]
        if X_c[i][int(feature_names)] < threshold:
            X_c[i] += [1]
        else:
            X_c[i] += [0]
    return X_c

class pltr():
    def __init__(self, penalization):
        self.penalization = penalization


    def fit(self, X, Y, X_test, max_depth=1):
        """
        Implementation of PLTR method
        :param X:
        :param Y:
        :param X_test:
        :param max_depth:
        :return:
        """
        X_thresh = deepcopy(X)
        X_thresh_test = deepcopy(X_test)
        p = np.array(X).shape[1]
        for k in range(p):
            clf = DecisionTreeClassifier(max_depth=max_depth).fit(np.array(X)[:, k].reshape(-1, 1), Y)
            X_thresh = tree_to_code(X_thresh, clf, k)
            X_thresh_test = tree_to_code(X_thresh_test, clf, k)
        if self.penalization > 0:
            self.clf = LogisticRegression(penalty="l1", solver="saga", max_iter=1000, C = self.penalization).fit(X_thresh, Y)
        else:
            self.clf = LogisticRegression(penalty="none",  max_iter=1000).fit(X_thresh, Y)
        self.X = X_thresh
        self.X_test = X_thresh_test

if __name__ == "__main__":
    Y, X, X_test, Y_test, betas, gammas, deltas = generate_default_hurlin(1000, 250, 5, False)
    clf = pltr()
    clf.fit(X, Y, X_test)
    print(clf.clf.score(clf.X_test, Y_test))

    logit = LogisticRegression().fit(X,Y)
    print(logit.score(X_test, Y_test))

