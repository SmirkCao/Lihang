#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: hmm
# Date: 9/17/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging
import warnings


class HMM(object):

    def __init__(self, n_component=0, Q=None, V=None):
        self.A = None
        self.B = None
        self.p = None
        self.M = 0
        self.N = n_component
        self.T = 0
        self.Q = Q
        self.V = V

    def _do_forward(self, X):
        prob = 0
        alpha = np.zeros((self.T, self.N))
        # A: NxM
        # B: NxM
        # alpha: TxN
        o = X[0]
        alpha[:, 0] = self.p * self.B[:, o]
        tmp = alpha[:, 0]
        for k, o in enumerate(X[1:]):
            alpha[:, k+1] = np.sum(tmp*self.A.T, axis=1)*self.B[:, o]
            if k+1 < len(X):
                tmp = alpha[:, k+1]
        # prob = np.log(np.sum(alpha[:, k+1]))
        prob = np.sum(alpha[:, k+1])
        return prob, alpha

    def _do_backward(self):
        pass

    def _do_estep(self):
        pass

    def _do_mstep(self):
        pass

    def fit(self, X):
        # 估计模型参数
        return self

    def predict(self, X):
        rst = None
        return rst

    def sample(self):
        rst = None
        return rst

    def score(self):
        rst = None
        return rst


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

