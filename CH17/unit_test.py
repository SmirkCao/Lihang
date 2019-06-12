#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 5/28/19
# Author: 😏 <smirk dot cao at gmail dot com>

# for vscode use
# Ref to : https://code.visualstudio.com/docs/python/unit-testing
import unittest
from sklearn import datasets
from lsa import LSA as lsa_test
from sklearn.decomposition import TruncatedSVD as lsa_sklearn
import matplotlib.pyplot as plt
import numpy as np
import sys


class TestLSAMethods(unittest.TestCase):
    def test_lsa_puffinwarellc_tutorial(self):
        #
        x = np.array([[0., 0., 1., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 1.],
                      [0., 1., 0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                      [1., 0., 0., 0., 0., 1., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                      [0., 0., 0., 0., 0., 2., 0., 0., 1.],
                      [1., 0., 1., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 1., 1., 0., 0., 0., 0.]])
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        
        print("\n")
        print(40*"*"+"u"+40*"*")
        print(np.round(u[:, :3], 2))

        print(40*"*"+"s"+40*"*")
        print(np.round(s[:3], 2))
        print(40*"*"+"vh"+40*"*")
        print(np.round(vh[:3, :], 2))

        print(40*"*"+"svh"+40*"*")
        print(np.round(np.dot(s[:3]*np.eye(3), vh[:3, :]), 2))

        # v based decision
        max_abs_raws = np.argmax(np.abs(vh), axis=1)
        signs = np.sign(vh[range(vh.shape[0]), max_abs_raws])
        u *= signs
        vh *= signs[:, np.newaxis]

        print("\n")
        print(40*"*"+"u"+40*"*")
        print(np.round(u[:, :3], 2))

        print(40*"*"+"s"+40*"*")
        print(np.round(s[:3], 2))
        print(40*"*"+"vh"+40*"*")
        print(np.round(vh[:3, :], 2))

        print(40*"*"+"svh"+40*"*")
        print(np.round(np.dot(s[:3]*np.eye(3), vh[:3, :]), 2))

    def test_lsa_fig_1701(self):
        x = np.array([[2., 0., 0., 0.],
                      [0., 2., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 2., 3.],
                      [0., 0., 0., 1.],
                      [1., 2., 2., 1.]])

        u, s, vh = np.linalg.svd(x, full_matrices=False)

        # v based decision
        max_abs_raws = np.argmax(np.abs(vh), axis=1)
        signs = np.sign(vh[range(vh.shape[0]), max_abs_raws])
        u *= signs
        vh *= signs[:, np.newaxis]
        
        print("\n")
        print(40*"*"+"u"+40*"*")
        print(np.round(u[:, :3], 2))
        print(40*"*"+"s"+40*"*")
        print(np.round(s[:3], 2))
        print(40*"*"+"vh"+40*"*")
        print(np.round(vh[:3, :], 2))

        print(40*"*"+"svh"+40*"*")
        print(np.round(np.dot(s[:3]*np.eye(3), vh[:3, :]), 2))

    def test_lsa_1701(self):
        base_path = sys.path[0]
        x = np.genfromtxt(base_path+"/data/data_1701.csv", delimiter=",")
        # print(x)
        lsa1 = lsa_sklearn(n_components=3)
        rst = lsa1.fit_transform(x)
        print("\n")
        print("singular_values\n", lsa1.singular_values_)
        print("components\n", lsa1.components_)
        print("rst\n", rst)

    def test_lsa(self):
        x = np.array([[0., 0., 1., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 1.],
                      [0., 1., 0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                      [1., 0., 0., 0., 0., 1., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                      [0., 0., 0., 0., 0., 2., 0., 0., 1.],
                      [1., 0., 1., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 1., 1., 0., 0., 0., 0.]])

        svd = lsa_sklearn(n_components=3, n_iter=7, random_state=42)
        svd.fit(x)
        print("\n")
        print("lsa sklearn")
        print("components_\n", svd.components_)
        print(svd.singular_values_)
        print(svd.explained_variance_)
        print(svd.explained_variance_ratio_)
        print(svd.explained_variance_ratio_.sum())

        svd_1 = lsa_test(n_components=3)
        rst = svd_1.fit(x)
        print("lsa test")
        print(svd_1.components)
        print(svd_1.singular_values)
        print(svd_1.explained_variance)
        print(svd_1.explained_variance_ratio)

        import utils
        labels = ["A", "B", "C", "D", "E", "F"][:3]
        feas = ["Book", "Dads", "Dummies", "Estate", "Guide", "Investing",
                "Market", "Real", "Rich", "Stock", "Value"]
        radar = utils.Radar(feas=feas, labels=labels)
        print(svd_1.u[:, :3].shape)
        radar.plot(svd_1.u[:, :3].T)

    def test_plot_radar(self):
        import utils
        import pandas as pd
        import sys

        base_path = sys.path[0]
        data = pd.read_csv(base_path+"/data/cities_ranking.csv")
        print(data.head())
        feas = ["A", "B", "C", "D", "E", "F"]
        radar = utils.Radar(feas=feas, labels=["SH", "BJ"])
        radar.plot(data[feas].values)