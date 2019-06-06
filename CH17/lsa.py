#! /usr/bin/env python
# Project:  Lihang
# Filename: lsa
# Date: 6/03/19
# Author: 😏 <smirk dot cao at gmail dot com>
# 截断奇异值分解用在count/tf-idf矩阵的时候，叫做潜在语义分析。


class LSA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, x):
        pass
