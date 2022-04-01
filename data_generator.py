import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from copy import deepcopy

def generate_default_times(intensity=0.2, censure_intensity=0.08, n_obs=100):
    times = np.random.exponential(1 / intensity, n_obs)
    censures = np.random.exponential(1 / censure_intensity, n_obs)
    return [min(times[k], censures[k]) for k in range(n_obs)], [1 if times[k] < censures[k] else 0 for k in
                                                                range(n_obs)]

def pltr(X,Y, X_test, Y_test, max_depth=2):
    X_thresh = deepcopy(X)
    X_test_thresh = deepcopy(X_test)

    clf = DecisionTreeClassifier(max_depth=max_depth).fit(X,Y)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        if not is_leaves[i]:
            for j in range(len(X)):
                X_thresh[j] += [1 if X[j][feature[i]] < threshold[i] else 0]
            for j in range(len(X_test)):
                X_test_thresh[j] += [1 if X_test[j][feature[i]] < threshold[i] else 0]
    ltr = LogisticRegression().fit(X_thresh,Y)
    score = ltr.score(X_test_thresh, Y_test)
    return ltr, score

def generate_default_hurlin(n_indiv=5000, n_indiv_test=2500, n_params=10, non_linear=True):
    X = [list(np.random.normal(0, 1, n_params + 1)) for _ in range(n_indiv)]
    X_test = [list(np.random.normal(0, 1, n_params + 1)) for _ in range(n_indiv_test)]
    lower_bonds = [sorted([X[i][k] for i in range(n_indiv)])[int(0.1 * n_indiv)] for k in range(n_params)]
    upper_bonds = [sorted([X[i][k] for i in range(n_indiv)])[int(0.9 * n_indiv)] for k in range(n_params)]
    betas = np.random.uniform(-1, 1, int(n_params * (n_params + 1) / 2))
    gammas = [np.random.uniform(lower_bonds[k], upper_bonds[k], 1) for k in range(n_params)]
    deltas = [np.random.uniform(lower_bonds[k], upper_bonds[k], 1) for k in range(n_params)]
    Y = [betas[0] +
         np.sum([betas[k + 1] if X[i][k] < gammas[k] else 0 for k in range(n_params)]) +
         np.sum([betas[k + j] if X[i][k] < deltas[k] and X[i][j] < deltas[j] else 0 for k in range(n_params) for j in
                 range(k, n_params)])
         for i in range(n_indiv)]
    Y_test = [betas[0] +
         np.sum([betas[k + 1] if X_test[i][k] < gammas[k] else 0 for k in range(n_params)]) +
         np.sum([betas[k + j] if X_test[i][k] < deltas[k] and X_test[i][j] < deltas[j] else 0 for k in range(n_params) for j in
                 range(k, n_params)])
         for i in range(n_indiv_test)]
    if non_linear:
        for i in range(n_indiv):
            X[i] += [X[i][k] ** 2 for k in range(n_params)] + [X[i][k] * X[i][j] for k in range(n_params) for j in
                                                               range(k, n_params)]
        for i in range(n_indiv_test):
            X_test[i] += [X_test[i][k] ** 2 for k in range(n_params)] + [X_test[i][k] * X_test[i][j] for k in range(n_params) for j in
                                                               range(k, n_params)]
    Y = [1 / (1 + np.exp(-Y[i])) for i in range(n_indiv)]
    Y_test = [1 / (1 + np.exp(-Y_test[i])) for i in range(n_indiv_test)]
    med = np.median(Y)
    med_test = np.median(Y_test)
    Y = [1 if Y[i] > med else 0 for i in range(n_indiv)]
    Y_test = [1 if Y_test[i] > med_test else 0 for i in range(n_indiv_test)]

    return Y, X, X_test, Y_test, betas, gammas

    