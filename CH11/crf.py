#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: crf
# Date: 9/21/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging


class CRF(object):
    def __init__(self, y=[]):
        """
        y: 标记的集合
        """
        self.alpha = None
        self.beta = None
        self.w_k = None
        self.f_k = None
        self.M = None
        self.K = 0
        self.y = y
        self.m = len(y)
        self.n = 0
        self.f = None
        self.Z = None

    def fit(self, X):
        self.n = len(X)
        self.w_k = np.zeros(self.K)

    def predict(self, X):
        _, seq = self.decode(X)
        return seq

    def predict_proba(self, X):
        pass

    def decode(self, X):
        # 类似序列标注
        prob = None
        seq = None
        return prob, seq

    def _f_gen(self):
        # self.f
        # from template to generate the feature function
        return self

    def _calc_f_k(self):
        self.M = np.array((self.m, self.m, self.K))

        # 实际上f_k应该是m*m*k的矩阵
        # i -> i-1
        # j -> i
        for k in range(self.K):
            for i, y_i in enumerate(self.y):
                for j, y_j in enumerate(self.y):
                    self.M[i, j, :] = self.f[k](y_i, y_j, i, j)

        return self

    def _calc_M(self):

        self.M = None
        self.M = np.exp(np.sum(self.w_k*self.f_k))
        return self

    def _do_forward(self, X):
        y0 = self.y[0]
        self.alpha[0] = y0
        for i in range(1, self.n+1):
            self.alpha[i] = self.alpha[i-1]*self.M[i]
        return self

    def _do_backward(self, X):
        yn1 = self.y[0]
        self.beta[-1] = yn1
        for i in range(self.n+1, 0, -1):
            self.beta[i] = self.M[i+1]*self.beta[i+1]
        return self

    def _virtebi(self, X):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
