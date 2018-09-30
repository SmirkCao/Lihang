# -*-coding:utf-8-*-
# Project:  Lihang
# Filename: unit_test
# Date: 8/15/18
# Author: 😏 <smirk dot cao at gmail dot com>
from knn import *
import numpy as np
import argparse
import logging
import unittest


class TestStringMethods(unittest.TestCase):

    def test_e31(self):
        X = np.loadtxt("Input/data_3-1.txt")
        # print(X-X[0])
        rst = np.linalg.norm(X - X[0], ord=1, axis=1)
        for p in range(2, 5):
            rst = np.vstack((rst, np.linalg.norm(X-X[0], ord=p, axis=1)))
        # Lp(x1,x2)
        self.assertListEqual(np.round(rst[:, 1], 2).tolist(), [4]*4)
        # Lp(x1,x3)
        self.assertListEqual(np.round(rst[:, 2], 2).tolist(), [6, 4.24, 3.78, 3.57])
        # print(np.round(rst[:, 2], 2).tolist())

    def test_e32(self):
        X = np.loadtxt("Input/data_3-2.txt")
        clf = KNN()
        clf.fit(X)
        logger.info(clf.kdtree)

    def test_e33(self):
        pass

    def test_q31(self):
        pass

    def test_q32(self):
        X = np.loadtxt("Input/data_3-2.txt")
        target = np.array([3, 4.5])
        clf = KNN()
        clf.fit(X)
        rst = clf.predict(target)
        self.assertListEqual([4, 7], rst.tolist())
        logger.info(rst)

    def test_q33(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()
