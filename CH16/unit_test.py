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
        x_star = x-np.mean(x, axis=1).reshape(-1, 1)
        # x_star = (x-np.mean(x, axis=1).reshape(-1, 1)) / \
        #     np.sqrt(np.var(x, axis=1)).reshape(-1, 1)
        print("means:\n", np.mean(x, axis=1))
        print("variances:\n", np.var(x, axis=1))
        print("x_star:\n", x_star)
        print("x_star means:\n", np.mean(x_star, axis=1))
        print("x_star variances:\n", np.var(x_star, axis=1))

        # x_ = x_star.T/np.sqrt(x_star.shape[0]-1)
        x_ = x_star.T
        u, s, vh = np.linalg.svd(x_, full_matrices=False)
        print("x_:\n", x_)
        print("u:\n", u)
        print("s:\n", s)
        print("vh:\n", vh)

        # full_matrices=True
        # u, s, vh = np.linalg.svd(x_)
        # s = s*np.eye(2)
        # y = np.dot(s, vh[:2, :])
        # rst = np.dot(u[:, :2], y)

        rst = np.dot(u*s, vh)
        print("rst: \n", rst)

        # 两个特征可以做可视化
        plt.figure(figsize=(5, 5))
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.grid()
        plt.scatter(x_star[0, :], x_star[1, :], label="raw data", alpha=0.3)
        rst = np.dot(vh, x_star)
        plt.scatter(rst[0, :], rst[1, :], label="svd based pca", alpha=0.3)

        # signs handling
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        vh_signs = vh*signs[:, np.newaxis]
        rst = np.dot(vh_signs, x_star)
        plt.scatter(rst[0, :], rst[1, :], marker="o", s=12, alpha=0.3,
                    label="svd based pca with signs")

        # for sklearn pca comparison
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        # pca in sklearn, (n_samples, n_features)
        rst = pca.fit_transform(x.T)
        print("sklearn pca:\n", rst)

        plt.scatter(rst[:, 0], rst[:, 1], label="sklearn_rst", alpha=0.3)
        plt.legend()

        axis_x = np.array([[0, 2],
                           [0, 0]])
        axis_y = np.array([[0, 0],
                           [0, 2]])
        plt.plot(axis_x[0, :], axis_x[1, :],
                 linestyle="-", c="red")
        plt.plot(axis_y[0, :], axis_y[1, :],
                 linestyle="-", c="blue")

        plt.plot(np.dot(vh, axis_x)[0, :], np.dot(vh, axis_x)[1, :],
                 linestyle="--", c="red", alpha=0.3)
        plt.plot(np.dot(vh, axis_y)[0, :], np.dot(vh, axis_y)[1, :],
                 linestyle="--", c="blue", alpha=0.3)

        plt.plot(np.dot(vh_signs, axis_x)[0, :],
                 np.dot(vh_signs, axis_x)[1, :],
                 linestyle="-.", c="red")
        plt.plot(np.dot(vh_signs, axis_y)[0, :],
                 np.dot(vh_signs, axis_y)[1, :],
                 linestyle="-.", c="blue")
        plt.savefig("ex1601.png")
        plt.show()

    def test_pca(self):
        """

        PCA分析
        """
        print("\n")
        # raw data from ex1601
        x = np.array([[2, 3, 3, 4, 5, 7],
                      [2, 4, 5, 5, 6, 8]])
        # 去掉均值
        x = x-np.mean(x, axis=1).reshape(-1, 1)
        print(x)
        assert (np.mean(x, axis=1) == np.zeros(2)).all()

        # for sklearn x.shape == (n_samples, n_features)
        pca_sklearn = skpca(n_components=2)
        pca_sklearn.fit(x.T)
        pca_sklearn_rst = pca_sklearn.fit_transform(x.T).T

        print("\n")
        print(40*"*"+"sklearn_pca"+40*"*")
        print("singular values:\n", pca_sklearn.singular_values_)
        print("explained variance ratio:\n",
              pca_sklearn.explained_variance_ratio_)
        print("transform:\n", )

        print(40*"*"+"smirk_pca"+40*"*")
        pca_test = smirkpca(n_components=2)
        pca_test_rst = pca_test.fit_transform(x)
        print("singular values:\n",
              pca_test.singular_values_)
        print("explained variance ratio:\n",
              pca_test.explained_variance_ratio_)
        print("transform:\n", pca_test_rst)

        self.assertTrue(np.allclose(pca_sklearn.singular_values_,
                                    pca_test.singular_values_))
        self.assertTrue(np.allclose(pca_sklearn_rst, pca_test_rst))
        self.assertTrue(np.allclose(pca_sklearn.explained_variance_ratio_,
                                    pca_test.explained_variance_ratio_))

    def test_pca_get_fig(self):
        pass
