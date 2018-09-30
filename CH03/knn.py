# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: knn
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>

# refs: https://en.wikipedia.org/wiki/K-d_tree

from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import numpy as np


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


class KNN(object):
    def __init__(self,
                 k=1,
                 p=2):
        """

        :param k: knn
        :param p:
        """
        self.k = k
        self.p = p
        self.kdtree = None

    @staticmethod
    def _fit(X, depth=0):
        try:
            k = X.shape[1]
        except IndexError as e:
            return None

        axis = depth % k
        X = X[X[:, axis].argsort()]
        median = X.shape[0] // 2  # choose median

        try:
            X[median]
        except IndexError:
            return None
        return Node(
            location=X[median],
            left_child=KNN._fit(X[:median], depth + 1),
            right_child=KNN._fit(X[median + 1:], depth + 1)
        )

    def _search(self, point):
        self.kdtree[0]

    def fit(self, X):
        self.kdtree = KNN._fit(X)
        return self.kdtree

    def predict(self, X):
        return[[2]]

    def predict_proba(self, X):
        pass


if __name__ == '__main__':
    pass
