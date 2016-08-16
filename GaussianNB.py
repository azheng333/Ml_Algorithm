# coding=utf-8
from sklearn import datasets
import numpy as np
import math


def gaussian(x_ij, mean, variance):
    exponent = math.exp(-(math.pow(x_ij - mean, 2) / (2 * variance)))
    return (1 / (math.sqrt(2 * math.pi * variance))) * exponent


def update_mean_variance(x_i):
    new_mu = np.mean(x_i, axis=0)
    new_var = np.var(x_i, axis=0)
    return new_mu, new_var


def probability_for_each_data(x, mean, variance):
    p = []
    for i in x:
        n_ij = 1
        for j, x_ij in enumerate(i):
            n_ij *= gaussian(x_ij, mean[j], variance[j])
        p.append(n_ij)
    return p


def probability_for_each_class(x, class_num, means, variances):
    probability = []
    for i in range(class_num):
        mean, variance = means[i, :], variances[i, :]
        probability.append(probability_for_each_data(x, mean, variance))
    return np.array(probability).T


def fit(x, y):
    classes = np.unique(y)
    means = np.zeros((len(classes), x.shape[1]))
    variances = np.zeros((len(classes), x.shape[1]))
    for y_i in classes:
        i = classes.searchsorted(y_i)
        x_i = x[y == y_i, :]
        new_mean, new_variance = update_mean_variance(x_i)
        means[i, :] = new_mean
        variances[i, :] = new_variance
    return means, variances


def predict(x, y, means, variances):
    classes = np.unique(y)
    probability = probability_for_each_class(x, len(classes), means, variances)
    print classes[np.argmax(probability, axis=1)]


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    means, variances = fit(x, y)
    predict(x, y, means, variances)

#------------------------------------
# from sklearn import datasets
# from sklearn.naive_bayes import GaussianNB
#
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
#
# gnb = GaussianNB()
# clf = gnb.fit(x, y)
# print clf.predict(iris.data)

