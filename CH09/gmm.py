#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: gmm
# Date: 9/5/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging
import warnings
import functools
import copy


class GMM(object):

    def __init__(self,
                 n_components=1,
                 tol=1e-3,
                 max_iter=100,
                 random_state=None,
                 verbose=0,
                 verbose_interval=10,
                 weight = None,
                 means = None,
                 covariances = None
                 ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.converged_ = False
        self.n_iter_ = 0
        self.weights_ = weight if weight is not None else np.random.rand(self.n_components)
        self.means_ = means if means is not None else np.random.rand(self.n_components)
        self.covariances_ = covariances if covariances is not None else np.random.rand(self.n_components)
        self.resp_ = None
        self.X_ = None
        self.sum0 = functools.partial(np.sum, axis=0)
        self.sum1 = functools.partial(np.sum, axis=1)
        if self.verbose:
            logger.info("Init: weights %s, means %s, covariances %s" % (self.weights_, self.means_, self.covariances_))

    def _gaussian_density(self, means, covariances, X=None):
        """
        Eq: 9.25
        :param X:
        :param means:
        :param covariances:
        :return:
        """
        if X is None:
            # rst = (1/np.sqrt(2*np.pi)/covariances)*np.exp(-(self.X_-means)**2/2/covariances/covariances)
            rst = (1/2/covariances)*np.exp(-(self.X_-means)**2/2/covariances/covariances)
        return rst

    def _e_step(self):
        sum1 = self.sum1

        resp = self.weights_*self._gaussian_density(means=self.means_, covariances=self.covariances_)
        Z = sum1(resp)
        resp = resp/np.reshape(np.tile(Z, self.n_components), (-1, 2), order="F")
        return resp

    def _m_step(self, resp):
        sum0 = self.sum0

        Z = sum0(resp)
        self.covariances_ = sum0(resp*(self.X_-self.means_)**2)/Z
        self.weights_ = Z/X.shape[0]
        self.means_ = sum0(resp*self.X_)/Z
        return self

    def fit(self, X):
        self.converged_ = False
        self.X_ = np.reshape(np.tile(X, self.n_components), (-1, self.n_components), order="F")

        for n_iter in range(self.max_iter):
            resp = self._e_step()
            self._m_step(resp)
            if self.resp_ is not None:
                delta = np.min(resp-self.resp_)
                if abs(delta) < self.tol:
                    self.converged_ = True
                    break
            self.resp_ = copy.deepcopy(resp)

        if not self.converged_:
            warnings.warn("Try different init parameters,"
                          "or increase max_iter, tol")

    def predict(self, X):
        labels = None
        return labels

    def predict_proba(self, X):
        pass


def get_dummy():
    np.random.seed(0)
    n_samples = 1000
    mu1 = 3
    mu2 = 4
    sigma1 = 0.1
    sigma2 = 0.3
    alpha1 = 0.3
    alpha2 = 0.7
    dummy_data = np.hstack([np.random.normal(mu1, sigma1, np.int64(alpha1*n_samples)),
                            np.random.normal(mu2, sigma2, np.int64(alpha2*n_samples))])
    logger.info("Dummy data: weights %s, means %s, covariances %s" % ([alpha1, alpha2], [mu1, mu2], [sigma1, sigma2]))
    return dummy_data


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())

    X = get_dummy()
    gmm = GMM(n_components=2, verbose=1)
    gmm.fit(X)
    logger.info("weights %s, means %s, covariances %s" % (gmm.weights_, gmm.means_, gmm.covariances_))