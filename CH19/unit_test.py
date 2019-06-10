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
from mcmc import Rejection, Accept
import matplotlib.pyplot as plt
import numpy as np


class TestMCMCMethod(unittest.TestCase):
    def test_exa_1901(self):
        pass

    def test_exa_1902(self):
        pass

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

        def q(x, mu=0.76, sigma=1, c=1.55):
            rst = c*1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
            return rst

        x = np.arange(-2, 4, 0.01)
        y = cpdf(x)
        ars = Rejection()
        ars.px = cpdf
        ars.qx = q
        rst = ars.sample(1000)
        act = Accept()
        act.sample(1000, 100)
        
        print(rst)
