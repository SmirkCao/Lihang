#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: pca
# Date: 5/31/19
# Author: 😏 <smirk dot cao at gmail dot com>
# from svd import SVD # someday
import numpy as np


class PCA(object):
    def __init__(self, n_components=2):
        self.n_components_ = n_components
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.u = None
        self.vh = None
        self.components_ = None

    def __str__(self,):
        rst = "PCA algorithms:\n"
        rst += "n_components: " + str(self.n_components_)
        return rst

    def fit(self, x):
        # check n_components and min(n_samples, n_features)
        n = x.shape[0]
        assert n > 1
        assert (np.mean(x, axis=1) == np.zeros(n)).all()
        x_ = x.T/np.sqrt(n-1)
        u, s, vh = np.linalg.svd(x_)
        self.vh = vh
        self.u = u
        self.singular_values_ = s
        self.explained_variance_ratio_ = s**2/np.sum(s**2)
        print(self.u)
        print(self.vh)

    def fit_transform(self, x):
        self.fit(x)
        self.components_ = np.dot(self.vh, x)
        return self.components_