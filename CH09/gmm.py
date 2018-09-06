#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: gmm
# Date: 9/5/18
# Author: 😏 <smirk dot cao at gmail dot com>
import argparse
import logging


class GMM(object):
    def __init__(self,
                 n_components=1,
                 tol=1e-3,
                 max_iter=100,
                 random_state=None,
                 verbose=0,
                 verbose_interval=10,
                 ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        labels = None
        return labels

    def predict_proba(self, X):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())