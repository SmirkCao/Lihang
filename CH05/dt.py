#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: dt
# Date: 10/5/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging
import warnings


class dt(object):
    def __init__(self,
                 tol=10e-3):
        self.tree = None
        self.tol = tol

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def _cal_entropy(y):
        pass

    @staticmethod
    def _cal_conditioanl_entropy(X, y):
        pass

    @staticmethod
    def _gain(X, y):
        return dt._cal_entropy(y) - dt._cal_conditioanl_entropy(X, y)

    @staticmethod
    def _gain_ratio(X, y):
        return dt._gain(X, y)/dt._cal_entropy(X)

    @staticmethod
    def _cal_gini(X, y):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
