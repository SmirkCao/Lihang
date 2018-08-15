# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: knn
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>

# refs: https://en.wikipedia.org/wiki/K-d_tree

from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


class KNN(object):
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.kdtree = None

    def lp_distance(self):
        print(self.k)
        return 1

    @staticmethod
    def _fit(point_list, depth=0):
        try:
            k = len(point_list[0])  # assumes all points have the same dimension
        except IndexError as e:  # if not point_list:
            return None
        # Select axis based on depth so that axis cycles through all valid values
        axis = depth % k

        # Sort point list and choose median as pivot element
        point_list.sort(key=itemgetter(axis))
        median = len(point_list) // 2  # choose median

        # Create node and construct subtrees
        return Node(
            location=point_list[median],
            left_child=KNN._fit(point_list[:median], depth + 1),
            right_child=KNN._fit(point_list[median + 1:], depth + 1)
        )

    def _search(self, point):
        self.kdtree[0]

    def fit(self, x_):
        self.kdtree = KNN._fit(x_)
        return self.kdtree

    def predict(self, x_):
        return[[2]]

    def predict_proba(self, x_):
        pass


if __name__ == '__main__':
    pass
