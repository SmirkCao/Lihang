#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 9/27/18
# Author: 😏 <smirk dot cao at gmail dot com>
from scipy import optimize
from svm import *
import matplotlib.pyplot as plt
import numpy as np
import sympy
import argparse
import logging
import unittest


class TestSVM(unittest.TestCase):
    skip_flag = True

    @unittest.skipIf(skip_flag, "debug")
    def test_e71(self):
        # data 2.1
        # x_1 = (3, 3), x_2 = (4, 3), x_3 = (1, 1)
        # ref: [example 16.3](http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf)
        fun = lambda x: ((x[0]) ** 2 + (x[1]) ** 2)/2
        cons = ({'type': 'ineq', 'fun': lambda x: 3 * x[0] + 3 * x[1] + x[2] - 1},
                {'type': 'ineq', 'fun': lambda x: 4 * x[0] + 3 * x[1] + x[2] - 1},
                {'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - x[2] - 1})
        res = optimize.minimize(fun, np.ones(3), method='SLSQP', constraints=cons)
        logger.info("\n res is \n %s \n x is \n %s\n" % (str(res), res["x"]))

        self.assertListEqual(res["x"].round(2).tolist(), [0.5, 0.5, -2])

        #  draw figure 7.4
        data = np.array([[3, 3],
                         [4, 3],
                         [1, 1]])
        figure, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], marker='o', edgecolors="b", c="w")

        (x1, x2) = sympy.symbols("x1,x2")

        w = res["x"]
        x_1 = np.array([x1, 0, 1])
        x_2 = np.array([0, x2, 1])
        y = np.array([-1, 0, 1])
        lines = list()
        for p in zip([(sympy.solve(e)[0], 0) for e in np.sum(w * x_1) + y],
                     [(0, sympy.solve(e)[0]) for e in np.sum(w * x_2) + y]):
            lines.append(p)
        # lines [H1, H, H2]

        for line_xs, line_ys in lines:
            ax.add_line(plt.Line2D(line_xs, line_ys, color="red"))

        plt.xlim([0, 8])
        plt.ylim([0, 8])
        plt.xlabel("$x^{(1)}$")
        plt.ylabel("$x^{(2)}$")
        plt.show()

    @unittest.skipIf(skip_flag, "debug")
    def test_e72(self):
        # data 2.1
        data = np.array([[3, 3],
                         [4, 3],
                         [1, 1]])
        label = np.array([1, 1, -1])

        def fun(a):
            return 4 * (a[0]) ** 2 + 13 / 2 * (a[1]) ** 2 + 10 * a[0] * a[1] - 2 * a[0] - 2 * a[1]
        # cons = ({'type': 'ineq', 'fun': lambda a: a[0]},
        #         {'type': 'ineq', 'fun': lambda a: a[1]})
        bnds = ((0, None), (0, None))
        res = optimize.minimize(fun, np.ones(2), method='SLSQP', bounds=bnds)

        alpha = res["x"]
        alpha = np.append(alpha, alpha[0] + alpha[1])
        w = np.sum((alpha * label).reshape(-1, 1) * data, axis=0)
        j = np.argmax(alpha)
        b = label[j] - np.sum(alpha * label * np.dot(data, data[j, :]), axis=0)
        #  0和2都OK，因为都为0.5 > 0
        # b = label[0] - np.sum(alpha*label*np.dot(data,data[0,:]),axis=0)
        # b = label[2] - np.sum(alpha*label*np.dot(data,data[2,:]),axis=0)
        self.assertListEqual(w.round(2).tolist(), [0.5, 0.5])
        self.assertEqual(b.round(2).tolist(), -2)
        logger.info("\nw is %s \n" % str(w.round(2)))
        logger.info("\nb is %s \n" % str(b.round(2)))

    @unittest.skipIf(skip_flag, "debug")
    def test_e73(self):
        logger.info("This ex is for introduce H and phi have not only one expression.")
        pass

    def test_e71_(self):
        # use this solver project
        # data 2.1
        data = np.array([[3, 3],
                         [4, 3],
                         [1, 1]])
        label = np.array([1, 1, -1])

        clf = SVM()
        clf.fit(data, label)
        print("test_e71_:", clf.alpha, clf.b)

    @unittest.skipIf(skip_flag, "debug")
    def test_e72_(self):
        # use this solver project
        # data 2.1
        data = np.array([[3, 3],
                         [4, 3],
                         [1, 1]])
        label = np.array([1, 1, -1])

        clf = SVM()
        clf.fit(data, label)
        print(clf.alpha)

    @unittest.skipIf(skip_flag, "debug")
    def test_mlia(self):
        # use dataset from mlia
        # load data
        # X = 0
        # clf = SVM()
        # clf.fit(X)
        # print(clf.alpha)
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()
