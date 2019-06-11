#! /usr/bin/env python
#!-*- coding=utf-8 -*-
# Project:  Lihang
# Filename: unit_test
# Date: 6/09/19
# Author: 😏 <smirk dot cao at gmail dot com>

# for vscode use
# Ref to : https://code.visualstudio.com/docs/python/unit-testing
import unittest
from sklearn import datasets
from mcmc import Rejection
import matplotlib.pyplot as plt
import numpy as np


class TestMCMCMethod(unittest.TestCase):
    def test_exa_1901(self):
        def f(x):
            return np.exp(-x**2/2)

        def p(x):
            return 1

        x = np.random.uniform(0, 1, 10)
        y = f(x)
        x_raw = np.arange(0, 1, 0.01)
        y_raw = f(x_raw)
        print(x.shape, y.shape)
        data = np.concatenate((x_raw[:, np.newaxis], y_raw[:, np.newaxis]),
                              axis=1)
        print(data)
        # data = data[data[:, 0].argsort()]
        rst = np.mean(y)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, marker='*')
        plt.plot(data[:, 0], data[:, 1], alpha=0.3)
        bbox = dict(boxstyle="round", fc="0.8")
        comment = r"$\int_0^1e^{-{x^2}/{2}}\mathrm{d}x=$"+str(round(rst, 2))
        plt.annotate(comment, (0.2, 0.8), bbox=bbox)
        plt.ylabel(r'$\exp\left(-\frac{1}{2}x^2\right)$')
        plt.ylim(-0.02, 1.02)
        plt.xlim(-0.02, 1.02)
        plt.show()

    def test_exa_1902(self):
        def p(x):
            return np.sqrt(2*np.pi)*np.exp(-x**2/2)

        def f(x):
            return x

        x = np.random.normal(0, 1, 10)
        y = f(x)
        x_raw = np.arange(-2, 2, 0.1)
        y_raw = f(x_raw)
        print(x.shape, y.shape)
        data = np.concatenate((x_raw[:, np.newaxis], y_raw[:, np.newaxis]),
                              axis=1)
        print(data)
        # data = data[data[:, 0].argsort()]
        rst = np.mean(y)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, marker='*', alpha=0.3)
        plt.plot(data[:, 0], data[:, 1], alpha=0.3)
        bbox = dict(boxstyle="round", fc="0.8")
        comment = r"$\int_{-\infty}^{\infty}x\frac{1}{\sqrt{2\pi}}e^{-{x^2}/{2}}\mathrm{d}x=$"+str(round(rst, 2))
        plt.annotate(comment, (-2, 2), bbox=bbox)

        plt.show()

    def test_exa_1903(self):
        pass

    def test_exa_1904(self):
        pass

    def test_exa_1905(self):
        pass

    def test_exa_1906(self):
        pass

    def test_exa_1907(self):
        pass

    def test_exa_1908(self):
        pass

    def test_uniform(self):

        print("\n", np.__version__)
        # 以下，1.16 及以下， numpy 1.17的接口变了
        s = np.random.uniform(-1, 0, 1000)
        print(np.all(s >= -1))
        print(np.all(s < 0))

        count, bins, ignored = plt.hist(s, 150, density=True)
        # draw PDF
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        plt.show()

    def test_normal(self):
        print("\n", np.__version__)
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 10000)
        print(abs(mu - np.mean(s)) < 0.01)
        print(abs(sigma - np.std(s, ddof=1)) < 0.01)

        count, bins, ignored = plt.hist(s, 300,
                                        density=True, label="sample hist")
        # draw PDF
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp(-(bins - mu)**2 / (2 * sigma**2)),
                 linewidth=2, color='r', label="pdf")
        plt.plot(bins, 2*(1/(sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-(bins - mu)**2 / (2 * sigma**2))),
                 linewidth=2, color='g', label="2*pdf")
        plt.plot(bins, 3*(1/(sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-(bins - mu)**2 / (2 * sigma**2))),
                 linewidth=2, color='b', label="3*pdf")
        plt.legend()
        plt.show()

    def test_cpdf(self):
        # customized probability density
        def cpdf(x, mu1=0.2, sigma1=0.5, mu2=1.6, sigma2=0.5, r=0.6):
            rst = r*1/(sigma1*np.sqrt(2*np.pi)) * \
                np.exp(-(x-mu1)**2/(2*sigma1**2))
            rst += (1-r)*1/(sigma2*np.sqrt(2*np.pi)) * \
                np.exp(-(x-mu2)**2/(2*sigma2**2))
            return rst

        mu, sigma, c = 0.76, 1, 1.55

        x = np.arange(-2, 4, 0.01)
        y = cpdf(x)
        plt.plot(x, y)
        plt.plot(x, c*1/(sigma*np.sqrt(2*np.pi))
                 * np.exp(-(x-mu)**2/(2*sigma**2)))
        plt.show()

    def test_alg_1901(self):
        # customized probability density
        def cpdf(x, mu1=0.2, sigma1=0.5, mu2=1.6, sigma2=0.5, r=0.6):
            rst = r*1/(sigma1*np.sqrt(2*np.pi)) * \
                np.exp(-(x-mu1)**2/(2*sigma1**2))
            rst += (1-r)*1/(sigma2*np.sqrt(2*np.pi)) * \
                np.exp(-(x-mu2)**2/(2*sigma2**2))
            return rst

        def q(x, mu=0.76, sigma=1):
            rst = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
            return rst

        x = np.arange(-2, 4, 0.01)
        y = cpdf(x)
        ars = Rejection(c=1.5)
        ars.px = cpdf
        ars.qx = q
        import time

        t0 = time.time()
        n_samples = 10000
        rst, rej_count = ars.sample(n_samples)
        t1 = time.time()
        # print("time: ", t1-t0)
        alpha = 0.7
        count, bins, ignored = plt.hist(rst, 150, density=True, alpha=alpha,
                                        label="sample hist", edgecolor="b")
        plt.plot(bins, cpdf(bins), color="r", linewidth=3, alpha=alpha,
                 label="p(x)")
        c = ars.c
        plt.plot(bins, c*q(bins), color="y", linewidth=3, alpha=alpha,
                 label=str(c)+"*q(x)")
        s = "Samples: {}\nRejects: {}\nTimes: {}".format(n_samples, rej_count, np.round(t1-t0, 2))
        plt.annotate(s, (np.min(bins), np.max(cpdf(bins))))
        plt.legend()
        plt.show()
