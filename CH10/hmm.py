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
        # todo: logsumexp trick
        alpha = np.zeros((self.N, self.T))
        # A: NxM
        # B: NxM
        # alpha: TxN
        o = X[0]
        alpha[:, 0] = self.p * self.B[:, o]
        tmp = alpha[:, 0]
        for k, o in enumerate(X[1:]):
            alpha[:, k+1] = np.sum(tmp*self.A.T, axis=1)*self.B[:, o]
            if k < len(X[1:]):
                tmp = alpha[:, k+1]
        # prob = np.log(np.sum(alpha[:, k+1]))
        prob = np.sum(alpha[:, k+1])
        return prob, alpha

    def _do_backward(self, X):
        beta = np.ones((self.N, self.T))
        o = X[-1]
        beta[:, -1] = 1
        tmp = beta[:, -1]
        # print(self.A, self.B, self.p, X)
        o_ = o
        for k, o in reversed(list(enumerate(X[:-1]))):
            beta[:, k] = np.sum(self.A*self.B[:, o_]*tmp, axis=1)
            if k > 0:
                tmp = beta[:, k]
                o_ = o
        prob = np.sum(self.p*self.B[:, o]*beta[:, 0])
        # print(beta, prob, prob, "new")
        return prob, beta

    def forward(self, obs_seq):
        """前向算法"""
        # 来源: https://applenob.github.io/hmm.html
        # F保存前向概率矩阵
        F = np.zeros((self.N, self.T))
        F[:, 0] = self.p * self.B[:, obs_seq[0]]

        for t in range(1, self.T):
            for n in range(self.N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]

        return F

    def backward(self, obs_seq):
        """后向算法"""
        # X保存后向概率矩阵
        # 来源: https://applenob.github.io/hmm.html
        X = np.zeros((self.N, self.T))
        X[:, -1:] = 1

        for t in reversed(range(self.T - 1)):
            X[:, t] = np.sum(self.A * self.B[:, obs_seq[t + 1]]*X[:, t + 1], axis=1)
        prob = np.sum(self.p * self.B[:, 0] * X[:, 0])
        print(prob, prob)
        return X

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

