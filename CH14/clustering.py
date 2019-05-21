#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: clustering
# Date: 5/15/19
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np


class Cluster(object):
    def __init__(self, name, data, linkage="single"):
        self.linkage = linkage
        self.name = name
        self.data = data[np.newaxis, :] if len(data.shape) == 1 else data
        assert len(self.data.shape) == 2

    def __sub__(a, b):
        distance = np.Inf
        if a.linkage == "single":
            for i in a.data:
                for j in b.data:
                    # dij
                    dij = np.sqrt(np.sum((i-j)**2))
                    if dij < distance:
                        distance = dij
        return distance

    def __or__(a, b):
        # rst = Cluster(a.name + "_" + b.name, 
        #               np.concatenate((a.data, b.data), axis=1))
        rst = Cluster(a.name + "_" + b.name, np.vstack((a.data, b.data)))
        return rst 

    def __str__(self, ):
        return "name: " + self.name + " data: " + str(self.data)


class Clustering(object):
    def __init__(self, k=2):
        self.labels = None
        self.k = k
        self.d = None
        self.metrics = None
        self.gs = None

    def fit(self, x):
        pass

    def predict(self, x):
        pass


class ClusterAgglomerative(Clustering):
    # algo 14.1
    def fit(self, x):
        n_samples = x.shape[0]
        self.labels = np.arange(n_samples)
        # assign cluster per sample
        gs = [Cluster(str(idx), data) for idx, data in enumerate(x)]
        print(len(gs), self.k)
        while len(gs) > self.k:
            mindistance = np.inf
            ga = None
            gb = None
            for g in gs:
                for g_ in gs:
                    if g == g_:
                        continue
                    distance = g - g_
                    if distance < mindistance:
                        ga = g
                        gb = g_
                        mindistance = distance
            if mindistance < np.inf:
                gs.remove(ga)
                gs.remove(gb)
                gs.append(ga | gb)
                # print(ga, gb, mindistance)
            # print("len of gs:", len(gs))
        self.gs = gs