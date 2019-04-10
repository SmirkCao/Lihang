# -*-coding:utf-8-*-
# Project: CH02
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from perceptron import *
import numpy as np
import argparse
import logging
import unittest


class TestPerceptron(unittest.TestCase):

    def test_e21(self):
        logger.info("test case e21")
        # data e2.1
        data_raw = np.loadtxt("Input/data_2-1.txt")
        X = data_raw[:, :2]
        y = data_raw[:, -1]
        clf = Perceptron(eta=1, verbose=False)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        logger.info(clf.w)
        logger.info(str(y_pred))
        self.assertListEqual(y.tolist(), y_pred.tolist())

    def test_e22(self):
        logger.info("test case e22")
        # data e2.1
        data_raw = np.loadtxt("Input/data_2-1.txt")
        X = data_raw[:, :2]
        y = data_raw[:, -1]
        clf = Perceptron(verbose=False)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        logger.info(clf.w)
        logger.info(str(y_pred))
        self.assertListEqual(y.tolist(), y_pred.tolist())

    def test_logic_1(self):
        # loaddata
        data_raw = np.loadtxt("Input/logic_data_1.txt")
        X = data_raw[:, :2]
        clf = Perceptron(max_iter=100, eta=0.0001, verbose=False)
        # test and
        y = data_raw[:, 2]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        logger.info("test case logic_1 and")
        self.assertListEqual(y.tolist(), y_pred.tolist())
        # test or
        logger.info("test logic_1 or")
        y = data_raw[:, 3]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        self.assertListEqual(y.tolist(), y_pred.tolist())
        # test not
        logger.info("test logic_1 not")
        y = data_raw[:, 4]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        self.assertListEqual(y.tolist(), y_pred.tolist())

    def test_logic_2(self):
        # loaddata
        data_raw = np.loadtxt("Input/logic_data_2.txt")
        X = data_raw[:, :3]
        clf = Perceptron(max_iter=100, eta=0.0001, verbose=False)
        # test and
        y = data_raw[:, 3]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        logger.info("test case logic_2 and")
        self.assertListEqual(y.tolist(), y_pred.tolist())
        # test or
        logger.info("test logic_2 or")
        y = data_raw[:, 4]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        self.assertListEqual(y.tolist(), y_pred.tolist())
        # test not
        logger.info("test logic_2 not")
        y = data_raw[:, 5]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        self.assertListEqual(y.tolist(), y_pred.tolist())

    def test_mnist(self):
        raw_data = load_digits(n_class=2)
        X = raw_data.data
        y = raw_data.target
        # 0和1比较容易分辨吧
        y[y == 0] = -1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)

        clf = Perceptron(verbose=False)
        clf.fit(X_train, y_train)
        test_predict = clf.predict(X_test)
        score = accuracy_score(y_test, test_predict)
        logger.info("The accruacy socre is %2.2f" % score)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

    unittest.main()
