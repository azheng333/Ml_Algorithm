from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn import datasets

X = [[2.3], [1.1]]
fit_X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

iris = datasets.load_iris()
fit_X = iris.data
y = iris.target
X = fit_X[0]

import operator

def predict(X, fit_X, y, k):
    dist = pairwise_distances(X, fit_X, 'euclidean')
    sorted_dist_index = dist.argsort()

    class_votes = []
    for j in xrange(sorted_dist_index.shape[0]):
        votes = {}
        for i in range(k):
            label = y[sorted_dist_index[j][i]]
            votes[label] = votes.get(label, 0) + 1
        class_votes.append(votes)

    sorted_class_votes = []
    for v in class_votes:
        sorted_vote = sorted(v.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_class_votes.append(sorted_vote[0][0])

    return sorted_class_votes

group, labels = np.array(fit_X), np.array(y)
print predict(np.array(X), group, labels, 3)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(fit_X, y)
print neigh.predict(X)
