# -*-coding:utf-8-*-
# Project: CH6  
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>


from logistic_regression import *
from maxent import *
from sympy import *
import numpy as np
import unittest
import argparse
import logging


class TestMEMethods(unittest.TestCase):

    def test_e61(self):
        # 1 constrains
        P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2", real=True)
        L = P1 * log(P1) + P2 * log(P2) + P3 * log(P3) + P4 * log(P4) + P5 * log(P5) \
            + w0 * (P1 + P2 + P3 + P4 + P5 - 1)
        P1_e = (solve(diff(L, P1), P1))[0]
        P2_e = (solve(diff(L, P2), P2))[0]
        P3_e = (solve(diff(L, P3), P3))[0]
        P4_e = (solve(diff(L, P4), P4))[0]
        P5_e = (solve(diff(L, P5), P5))[0]
        L = L.subs({P1: P1_e, P2: P2_e, P3: P3_e, P4: P4_e, P5: P5_e})
        w = (solve([diff(L, w0)], [w0]))[0]
        P = [P1_e.subs({w0: w[0]}),
             P2_e.subs({w0: w[0]}),
             P3_e.subs({w0: w[0]}),
             P4_e.subs({w0: w[0]}),
             P5_e.subs({w0: w[0]})]
        self.assertEqual([round(p, 5) for p in P], [round(p, 5) for p in [1/5, 1/5, 1/5, 1/5, 1/5]])

    def test_e62(self):
        # 2 constrains
        P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2", real=True)
        L = P1 * log(P1) + P2 * log(P2) + P3 * log(P3) + P4 * log(P4) + P5 * log(P5) \
            + w1 * (P1 + P2 - 3 / 10) \
            + w0 * (P1 + P2 + P3 + P4 + P5 - 1)
        P1_e = (solve(diff(L, P1), P1))[0]
        P2_e = (solve(diff(L, P2), P2))[0]
        P3_e = (solve(diff(L, P3), P3))[0]
        P4_e = (solve(diff(L, P4), P4))[0]
        P5_e = (solve(diff(L, P5), P5))[0]
        L = L.subs({P1: P1_e, P2: P2_e, P3: P3_e, P4: P4_e, P5: P5_e})
        w = (solve([diff(L, w1), diff(L, w0)], [w0, w1]))[0]
        P = [P1_e.subs({w0: w[0], w1: w[1]}),
             P2_e.subs({w0: w[0], w1: w[1]}),
             P3_e.subs({w0: w[0], w1: w[1]}),
             P4_e.subs({w0: w[0], w1: w[1]}),
             P5_e.subs({w0: w[0], w1: w[1]})]
        self.assertEqual([round(p, 5) for p in P], [round(p, 5) for p in [3/20, 3/20, 7/30, 7/30, 7/30]])

    def test_load_data(self):
        X, y = load_data('./Input/train_10.csv')
        print(X.shape, y.shape)
        return X, y

    def test_gradient_decent(self):
        x, y = self.test_load_data()

        x = x[:100]
        y = y[:100]
        print(x.shape, y.shape)

        # R2 loss
        # def f(x_, y_, w_):
        #     # $L(w)=\frac{1}{2N}\sum_{i=1}^{N}(y-w\cdot x)^2$
        #     rst = 0
        #     m = y_.size
        #     rst = 1 / (2 * m) * np.sum((y_ - np.dot(x_, w_)) ** 2)
        #     return rst
        #
        # def g(x_, y_, w_):
        #     # $L'(w)=\frac{1}{2}x\cdot(w\cdot x-y)$
        #     # rst = []
        #     m = y_.size
        #     rst = 1 / m * np.dot(x_.T, (np.dot(x_, w_) - y_))
        #     return rst

        # LR loss
        # def f(x_, y_, w_):
        #     # Logistic Regression Loss
        #     m = y_.size
        #     rst_ = -(1 / m) * np.sum(np.dot(x_, w_) * y_ - np.log(1 + np.exp(np.dot(x_, w_))))
        #     return rst_
        #
        # def g(x_, y_, w_):
        #     m = y_.size
        #     rst_ = (1 / m) * np.dot(x_.T, y_ * sigmoid(np.dot(x_, w_)))
        #     return rst_
        #
        # def sigmoid(x_):
        #     p = np.exp(x_)
        #     p = p / (1 + p)
        #     return p

        clf = LogisticRegression()
        clf.f = f
        clf.g = g
        rst_w, rst_cols = clf.gradient_descent(x, y)

        # coef_.shape is (10, 784)
        time_1 = time.time()
        rst = np.array([rst_cols[idx] for idx in [np.argmax(rst) for rst in 1-sigmoid(np.dot(x, rst_w.T))]])
        rst = np.vstack([rst, y])
        time_2 = time.time()
        print('predict cost ', time_2 - time_1, ' second', '\n')
        print(rst.T)

    def test_lr(self):
        x, y = self.test_load_data()

        x = x[:100]
        y = y[:100]
        logger.info("%s, %s" % (x.shape, y.shape))

        clf = LogisticRegression()
        clf.f = f
        clf.g = g
        clf.fit(x, y)
        rst = clf.predict(x,)
        logger.info(rst)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
    unittest.main()
