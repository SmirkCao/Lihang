#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: svm
# Date: 9/27/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging


class SVM(object):
    def __init__(self,
                 tol=10e-3,
                 C=0.6,
                 n_iters=10,
                 verbose=True):
        self.alpha = None
        self.b = 0
        self.tol = tol
        self.C = C
        self.n_iters = n_iters
        self.m = 0
        self.verbose = verbose

    def fit(self, X, y):
        self.m = len(X)
        self.alpha = np.zeros(self.m)
        self.b = 0
        self._do_smo(X, y)

    def predict(self, X):
        pass

    def predict_preba(self, X):
        pass

    def _do_smo(self, X, y):
        n_iter = 0
        while n_iter < self.n_iters:
            alpha_pairs_changed = 0
            for i in range(self.m):
                ei = self._do_ei(X, y, i)
                if ((y[i] * ei < -self.tol) and (self.alpha[i] < self.C)) or \
                   ((y[i] * ei > self.tol) and (self.alpha[i] > 0)):
                    j = self._do_selectj(i, self.m)
                    ej = self._do_ei(X, y, j)
                    alphaiold = self.alpha[i].copy()
                    alphajold = self.alpha[j].copy()
                    if y[i] != y[j]:
                        L = max(0.0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0.0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    if L == H:
                        if self.verbose:
                            print("L==H")
                        continue
                    eta = self._do_eta(X, i, j)
                    # 简化处理
                    if eta >= 0:
                        if self.verbose:
                            print("eta>=0")
                        continue
                    # alpha[j]
                    self.alpha[j] -= y[j] * (ei - ej) / eta
                    self.alpha[j] = self._do_clipalpha(self.alpha[j], H, L)

                    if abs(self.alpha[j] - alphajold) < 0.00001:
                        if self.verbose:
                            print("j not moving enough")
                        continue
                    # alpha[i]
                    self.alpha[i] += y[j] * y[i] * (alphajold - self.alpha[j])
                    #
                    b1 = self.b - ei - y[i] * (self.alpha[i] - alphaiold) * np.dot(X[i, :], X[i, :]) - \
                         y[j] * (self.alpha[j] - alphajold) * np.dot(X[i, :], X[j, :])

                    b2 = self.b - ej - y[i] * (self.alpha[i] - alphaiold) * np.dot(X[i, :], X[j, :]) - \
                         y[j] * (self.alpha[j] - alphajold) * np.dot(X[j, :], X[j, :])

                    if (0 < self.alpha[i]) and (self.C > self.alpha[j]):
                        self.b = b1
                    elif (0 < self.alpha[j]) and (self.C > self.alpha[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    alpha_pairs_changed += 1
                    if self.verbose:
                        print("iter: %d i: %d, paris changed %d" % (n_iter, i, alpha_pairs_changed))
            if alpha_pairs_changed == 0:
                n_iter += 1
            else:
                n_iter = 0
            if self.verbose:
                print("iteration number: %d" % n_iter)
        return self.alpha, self.b

    def _do_smop(self):
        return self.alpha, self.b

    def _do_gxi(self, X, y, i):
        gxi = np.sum(self.alpha*y*np.dot(X, X[i, :]), axis=0) + self.b
        return gxi

    def _do_ei(self, X, y, i):
        ei = self._do_gxi(X, y, i) - y[i]
        return ei

    @staticmethod
    def _do_eta(X, i, j):
        eta = 2 * np.dot(X[i, :], X[j, :]) - np.dot(X[i, :], X[i, :]) - np.dot(X[j, :], X[j, :])
        return eta

    @staticmethod
    def _do_selectj(i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    @staticmethod
    # todo: use numpy clip
    def _do_clipalpha(alpha, H, L):
        if alpha > H:
            alpha = H
        if L > alpha:
            alpha = L
        return alpha


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())
