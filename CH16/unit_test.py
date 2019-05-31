#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 5/25/19
# Author: 😏 <smirk dot cao at gmail dot com>

# for vscode use
# Ref to : https://code.visualstudio.com/docs/python/unit-testing
import unittest
from sklearn import datasets
from pca import PCA as smirkpca
from sklearn.decomposition import PCA as skpca
import matplotlib.pyplot as plt
import numpy as np


class TestPCAMethods(unittest.TestCase):
    def test_e_16_1(self):
        r = np.array([[1, 0.44, 0.29, 0.33],
                      [0.44, 1, 0.35, 0.32],
                      [0.29, 0.35, 1, 0.60],
                      [0.33, 0.32, 0.60, 1]])
        ev, sigma = np.linalg.eig(r)
        
        print("\n")
        print(40*"*"+"Engine Values"+40*"*")
        print(ev)

        print(40*"*"+"eta"+40*"*")
        denominator = np.sum(ev)
        for numerator in ev:
            print(np.round(numerator/denominator, 3))

        for idx in range(ev.shape[0]):
            print("engine value: ", np.round(ev[idx], 3))
            print("engine vector: ", np.round(sigma[:, idx], 3))
            print("factor loading, rho(xi,yj): ",
                  np.round(np.sqrt(ev[idx])*sigma[:, idx], 3))
            print("factor loading sum: ",
                  np.round(np.sum(ev[idx]*sigma[:, idx]**2), 3))
        print(40*"*"+"svd"+40*"*")
        u, s, vh = np.linalg.svd(r)
        print(u)
        print(s)
        print(vh)
        # s 特征值， vh 特征向量
    
    def test_ex1601(self):
        # raw data
        x = np.array([[2, 3, 3, 4, 5, 7],
                      [2, 4, 5, 5, 6, 8]])
                
        # normalization
        x_star = (x-np.mean(x, axis=1).reshape(-1, 1))/np.sqrt(np.var(x, axis=1)).reshape(-1, 1)
        print(np.mean(x, axis=1))
        print(np.var(x, axis=1))

        print(x_star)

        print(np.var(x_star, axis=1))
        x_ = x_star.T/np.sqrt(x_star.shape[0]-1)
        u, s, vh = np.linalg.svd(x_)
        print(x_)
        print("\n")
        print(u)
        print(s)
        print(vh)
        s = s*np.eye(2)
        # print(vh[:2, :].shape)
        y = np.dot(s, vh[:2, :])
        rst = np.dot(u[:, :2], y)
        print(rst)

        # s engine value
        # vh engine vector

        # 两个特征可以做可视化
        plt.figure(figsize=(5, 5))
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.grid()
        plt.scatter(x_star[0, :], x_star[1, :])
        plt.scatter(vh[:, 0], vh[:, 1])

        # print(rst.shape)
        # plt.scatter(vh[:, 0]*x_star[0, :], vh[:, 1]*x_star[1, :])
        plt.plot([0, 0], [1, 0.5], c="black", marker="*")
        plt.plot([0.5, 1], [0, 0], c="black", marker="*")
        a = np.array([[0.0, 0.0],
                      [0.5, 1.0]])
        b = np.array([[0.7, 1.5],
                      [0.0, 0.0]])
        rst = np.dot(vh, a)
        print(rst)
        plt.plot(rst[:, 0],
                 rst[:, 1], c="red", marker="*")

        rst = np.dot(vh, b)
        print(rst)
        plt.plot(rst[:, 0],
                 rst[:, 1], c="red", marker="*")

        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # # pca in sklearn, (n_samples, n_features)
        # rst = pca.fit_transform(x.T)
        # print(rst)

        # plt.scatter(rst[:, 0], rst[:, 1])
        # plt.show()

    def test_pca(self):
        # raw data
        x = np.array([[2, 3, 3, 4, 5, 7],
                      [2, 4, 5, 5, 6, 8]])
        # for sklearn x.shape == (n_samples, n_features)
        pca_sklearn = skpca(n_components=2)
        pca_sklearn.fit(x.T)
        print("\n")
        print(40*"*"+"sklearn_pca"+40*"*")
        print(pca_sklearn.explained_variance_ratio_)
        print(pca_sklearn.singular_values_)

        print(40*"*"+"smirk_pca"+40*"*")
        pca_test = smirkpca(n_components=2)
        print(pca_test)

    def test_pca_get_fig(self):
        pass
