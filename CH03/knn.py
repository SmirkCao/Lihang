# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: knn
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>

# refs: https://en.wikipedia.org/wiki/K-d_tree

from collections import namedtuple
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
        # todo: 这里可以展开，通过方差选择
        axis = depth % k
        X = X[X[:, axis].argsort()]
        median = X.shape[0] // 2

        try:
            X[median]
        except IndexError:
            return None

        return Node(
            location=X[median],
            left_child=KNN._fit(X[:median], depth + 1),
            right_child=KNN._fit(X[median + 1:], depth + 1)
        )

    def _distance(self, x, y):
        return np.linalg.norm(x-y, ord=self.p)

    def _search(self, point, tree=None, depth=0, best=None):
        if tree is None:
            return best

        k = point.shape[0]
        # update best
        if best is None or self._distance(point, tree.location) < self._distance(best, tree.location):
            next_best = tree.location
        else:
            next_best = best

        # update branch
        if point[depth%k] < tree.location[depth%k]:
            next_branch = tree.left_child
        else:
            next_branch = tree.right_child
        return self._search(point, tree=next_branch, depth=depth+1, best=next_best)

    def fit(self, X):
        self.kdtree = KNN._fit(X)
        return self.kdtree

    def predict(self, X):
        rst = self._search(X, self.kdtree)
        return rst

    def predict_proba(self, X):
        pass


if __name__ == '__main__':
    pass
