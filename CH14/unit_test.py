#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 5/15/19
# Author: 😏 <smirk dot cao at gmail dot com>

# for vscode use
# Ref to : https://code.visualstudio.com/docs/python/unit-testing
import unittest
from clustering import ClusterAgglomerative, Cluster, ClusterKmeans
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


class TestClusteringMethods(unittest.TestCase):
    def test_e14_2(self):
        data = [np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]]), 
                np.array([0, 1, 1, 1, 0])]
        clustering = ClusterAgglomerative(k=2)
        clustering.fit(data[0])
        print(data[0]),
        print(data[1])
        print(clustering.gs[0].data)
        print(clustering.gs[1].data)
        print(clustering.gs[0].name)
        print(clustering.gs[1].name)

        plt.scatter(clustering.gs[0].data[:, 0],
                    clustering.gs[0].data[:, 1])
        plt.scatter(clustering.gs[1].data[:, 0],
                    clustering.gs[1].data[:, 1])
        plt.show()

    def test_fit(self):
        n_samples = 150
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        clustering = ClusterAgglomerative(k=2)
        clustering.fit(data[0])
        print(data[0]),
        print(data[1])
        print(clustering.gs[0].data)
        print(clustering.gs[1].data)
        print(clustering.gs[0].name)
        print(clustering.gs[1].name)

        plt.scatter(clustering.gs[0].data[:, 0],
                    clustering.gs[0].data[:, 1])
        plt.scatter(clustering.gs[1].data[:, 0],
                    clustering.gs[1].data[:, 1])
        plt.show()

    def test_cluster(self):
        a = Cluster("a", np.array([[2], [2], [4]]))
        b = Cluster("b", np.array([[4], [5], [6]]))
        self.assertEqual(a-b, 0)
        print(a | b | b | a | b)

    def test_kmeans_blob(self):
        n_samples = 150
        data = datasets.make_blobs(n_samples=n_samples, centers=2, n_features=2)
        clustering = ClusterKmeans(k=2)
        clustering.fit(data[0])
        plt.scatter(clustering.gs[0].data[:, 0],
                    clustering.gs[0].data[:, 1])
        plt.scatter(clustering.gs[1].data[:, 0],
                    clustering.gs[1].data[:, 1])
        plt.show()
    
    def test_kmeans_circle(self):
        n_samples = 150
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        clustering = ClusterKmeans(k=2)
        clustering.fit(data[0])
        plt.scatter(clustering.gs[0].data[:, 0],
                    clustering.gs[0].data[:, 1])
        plt.scatter(clustering.gs[1].data[:, 0],
                    clustering.gs[1].data[:, 1])
        plt.show()

if __name__ == "main":
    unittest.main()
