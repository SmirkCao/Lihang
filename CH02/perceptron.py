# -*-coding:utf-8-*-
# Project: CH02
# Filename: perceptron
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import random
import argparse
import logging


class Perceptron(object):
    def __init__(self,
                 max_iter=5000,
                 eta=0.00001,
                 verbose=True):
        self.eta_ = eta
        self.max_iter_ = max_iter
        self.w = 0
        self.verbose = verbose

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1] + 1)
        correct_count = 0
        n_iter_ = 0

        while n_iter_ < self.max_iter_:
            index = random.randint(0, y.shape[0] - 1)
            xx_ = np.hstack([X[index], 1])
            yy_ = 2 * y[index] - 1
            wx = np.dot(self.w, xx_)

            if wx * yy_ > 0:
                correct_count += 1
                if correct_count > self.max_iter_:
                    break
                continue

            self.w += self.eta_ * yy_ * xx_
            n_iter_ += 1
            if self.verbose:
                print(n_iter_)

    def predict(self, X):
        # for b
        X = np.hstack([X, np.ones(X.shape[0]).reshape((-1, 1))])
        # activation function for perceptron: sign
        rst = np.array([1 if rst else -1 for rst in np.dot(X, self.w) > 0])
        # np.sign(0) == 0
        # rst = np.sign(np.dot(X, self.w))
        return rst


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
