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

    def __str__(self,):
        rst = "PCA algorithms:\n"
        rst += "n_components: " + str(self.n_components_)
        return rst
    
    def fit(self, x):
        # check n_components and min(n_samples, n_features)
        pass
    
    def fit_transform(x):
        return x