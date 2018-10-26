#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: bmm
# Date: 10/24/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging
"""
方便理解, 书中用到的符号列举在下面
N
i
j
"""


class BMM(object):
    def __init__(self,
                 n_components=2,
                 max_iter=100,
                 tol=1e-3):
        # k
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        # N
        self.m = 0
        self.n = 0
        self.gamma = None
        self.n_iter_ = 0
        self.mu = None
        self.alpha = None
        self.X = None
        self.label = None

    def fit(self,
            X,
            y=None):
        k = self.n_components
        # self.m, self.n = X.shape
        self.m = X.shape[0]
        N = self.m
        # 如果n > 1 , 分解成n个模型训练, 结果再拼回来
        if y is not None:
            self.label = y
        # gamma: (N, k), 样本对子模型的响应度gamma_jk, 按j求和应该是1
        self.gamma = np.ones((N, k))/k
        # alpha: (k) , 子模型对混合模型的贡献, 求和为1
        self.alpha = np.ones(k)/k
        # self.alpha = np.array([0.4, 0.6])
        mu = np.ones(k)/k
        # mu = np.array([0.6, 0.7])
        # mu: (k, 2) 2是为了做矩阵乘法, 相对for loop效率应该会高, 这里todo: benchmark
        self.mu = np.stack((mu, 1-mu), axis=-1)
        # X: (N, 2)
        self.X = np.eye(2)[X]

        for i in range(self.max_iter):
            print(self.alpha, "\n", self.mu[:, 0], "\n", self.gamma, "\n")
            self.do_e_step()
            self.do_m_step()
            if self.is_convergence():
                break

    def is_convergence(self):

        return False

    def density(self):
        # Bernoulli
        rst = np.dot(self.X, self.mu.T)
        return rst

    def do_e_step(self):
        # 更新gamma
        self.gamma = self.density()*self.alpha
        z = np.sum(self.gamma, axis=1).reshape(-1, 1)
        self.gamma /= z
        return self

    def do_m_step(self):
        nk = np.sum(self.gamma, axis=0).reshape(-1, 1)
        # update mu
        self.mu = np.dot(self.X[:, 1], self.gamma)/nk
        # update alpha
        self.alpha = (nk/self.m).reshape(2)
        return self

    def predict(self, X):
        pass

    def sample(self,
               n_samples=1):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())
