#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: dt
# Date: 10/5/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
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
        if y.shape[0] == 0:
            return 0
        unique, cnts = np.unique(y, return_counts=True)
        freq = cnts/y.shape[0]
        return -np.sum(freq*np.log2(freq))

    @staticmethod
    def _cal_conditioanl_entropy(X, y):
        if X.shape[0] == 0 or y.shape[0] == 0:
            return 0
        rst = 0
        items, cnts = np.unique(X, return_counts=True)
        for item, cnt in zip(items, cnts):
            ent = dt._cal_entropy(y[X == item])
            freq = cnt/X.shape[0]
            rst += freq*ent
        return rst

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
