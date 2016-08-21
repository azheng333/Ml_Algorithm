# coding=utf-8
import numpy as np
import random
from sklearn import datasets


def no_centers_move(centers_old, centers_new):
    return set([tuple(a) for a in centers_new]) == set([tuple(a) for a in centers_old])


def cluster_points(x, centers):
    clusters = {}
    for v in x:
        index = min([(i[0], np.linalg.norm(v - centers[i[0]])) for i in enumerate(centers)], key=lambda t: t[1])[0]
        try:
            clusters[index].append(v)
        except KeyError:
            clusters[index] = [v]
    return clusters


def new_centers(clusters):
    centers = []
    keys = sorted(clusters.keys())
    for k in keys:
        centers.append(np.mean(clusters[k], axis=0))
    return centers


def find_centers(x, n_clusters, centers):
    centers_old = random.sample(x, n_clusters)
    centers_new = centers
    clusters = {}
    while not no_centers_move(centers_old, centers_new):
        centers_old = centers_new
        clusters = cluster_points(x, centers_new)
        centers_new = new_centers(clusters)
    return centers_new, clusters


def distance_to_centers(x, centers):
    return np.array([min([np.linalg.norm(v - c) ** 2 for c in centers]) for v in x])


def next_center(x, distances):
    likelihood = distances / distances.sum()
    cumsum_likelihood = likelihood.cumsum()
    index = np.where(cumsum_likelihood >= random.random())[0][0]

    return x[index]


def init_centers(x, n_clusters):
    centers = random.sample(x, 1)
    while len(centers) < n_clusters:
        distances = distance_to_centers(x, centers)
        centers.append(next_center(x, distances))
    return centers


def fit(x):
    centers = init_centers(x, 3)
    centers, clusters = find_centers(x, 3, centers)


if __name__ == '__main__':
    iris = datasets.load_iris()
    fit(iris.data)



