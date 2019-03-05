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

    def __init__(self, n_component=0,
                 Q=None,
                 V=None,
                 n_iters=5):
        self.A = None
        self.B = None
        self.p = None
        self.M = 0
        self.N = n_component
        self.T = 0
        self.Q = Q
        self.V = V
        self.n_iters = n_iters
        self.alpha = None
        self.beta = None
        self.deta = None
        self.gamma = None
        self.xi = None
        self.Ei = None
        self.Ei_ = None
        self.Ei_j = None

    def init_param(self, X):
        # 最简单的初始化应该是均匀分布
        # 另外的方法是Dirichlet Distribution
        # todo: update Dirchlet Distribution
        if self.V is not None:
            self.M = len(self.V)
        else:
            warnings.warn("M warning")
        self.A = np.ones((self.N, self.N))/self.N
        self.B = np.ones((self.N, self.M))/self.M
        self.p = np.ones(self.N)/self.N
        self.T = len(X)
        return self

    def _do_forward(self, X):
        # todo: logsumexp trick
        alpha = np.zeros((self.N, self.T))
        # A: NxM
        # B: NxM
        # alpha: NxT
        t = 0
        o = X[t]
        alpha[:, t] = self.p * self.B[:, o]
        t_rest = np.arange(1, self.T)
        for t in t_rest:
            o = X[t]
            alpha[:, t] = np.sum(alpha[:, t-1]*self.A.T, axis=1)*self.B[:, o]

        self.alpha = alpha
        prob = np.sum(alpha[:, -1])
        return prob, alpha

    def _do_backward(self, X):
        beta = np.ones((self.N, self.T))

        t = -1
        beta[:, t] = 1
        # print(self.A, self.B, self.p, X)

        t_rest = np.arange(self.T-1)[::-1]
        for t in t_rest:
            o = X[t+1]
            beta[:, t] = np.sum(self.A*self.B[:, o]*beta[:, t+1], axis=1)
        self.beta = beta

        prob = np.sum(self.p*self.B[:, X[0]]*beta[:, 0])
        # print(beta, prob, prob, "new")
        return prob, beta

    # 后面这两个主要是为了验证前向后向的结果
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
        # print(prob)
        return X

    def _do_estep(self, X):
        # 在hmmlearn里面是会没有专门的estep的
        _, self.alpha = self._do_forward(X)
        _, self.beta = self._do_backward(X)
        post_prior = self.alpha*self.beta
        # Eq. 10.24
        self.gamma = post_prior/np.sum(post_prior)
        # Eq. 10.26
        left_a = self.alpha
        right_a = np.dot(self.B, np.eye(len(X))[X, :len(set(X))].T)*self.beta
        trans_post_prior = np.array([x*self.A*y for x, y in zip(left_a[:, :-1].T, right_a[:, 1:].T)])
        self.xi = trans_post_prior/np.sum(trans_post_prior)
        # Eq. 10.27
        self.Ei = np.sum(self.gamma, axis=1)
        # Eq. 10.28
        self.Ei_ = np.sum(self.gamma[:, :-1], axis=1)
        # Eq. 10.29
        self.Ei_j = np.sum(self.xi[:, :, :-1], axis=2)
        return self

    def _do_mstep(self, X):
        # Eq. 10.39
        self.A = self.Ei_j/self.Ei

        # Eq. 10.40
        gamma_o = np.array([np.outer(x, y) for x, y in zip(self.gamma.T, np.eye(len(X))[X, :len(set(X))].T)])
        self.B = np.sum(gamma_o, axis=2).T/self.Ei.reshape(-1, 1)

        # Eq. 10.41
        self.p = self.gamma[:, 0]
        return self

    def fit(self, X):
        # 估计模型参数
        self.init_param(X)
        for n_iter in range(self.n_iters):
            self._do_estep(X)
            self._do_mstep(X)
            # convergence check
        #    if False:
        #        return rst
        return self

    def decode(self, X):
        """
        Find most likely state sequence corresponding to ``X``.
        """
        if self.T == 0:
            warnings.warn("T warning")
        if self.N == 0:
            warnings.warn("N warning")

        hidden_states = np.zeros(self.T)
        delta = np.ones((self.N, self.T))
        psi = np.zeros((self.N, self.T))

        t = 0
        o = X[t]
        delta[:, t] = self.p*self.B[:, o]
        psi[:, t] = 0
        t_rest = np.arange(1, self.T)
        for t in t_rest:
            o = X[t]
            delta[:, t] = np.max(delta[:, t-1]*self.A.T, axis=1)*self.B[:, o]
            psi[:, t] = np.argmax(delta[:, t-1]*self.A.T, axis=1)

        self.delta = delta
        prob = np.max(delta[:, -1])
        hidden_states[-1] = np.argmax(delta[:, -1])
        # t in T-1,...,1
        t_rest = np.arange(self.T-1)[::-1]
        for t in t_rest:
            hidden_states[t] = np.argmax(delta[:, t]*self.A[:, int(hidden_states[t+1])], axis=0)

        return prob, hidden_states

    def predict(self, X):
        """
        Find most likely state sequence corresponding to ``X``.
        """
        _, states = self.decode(X)
        return states

    def predict_proba(self):
        post_prior = 0

        return post_prior

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

