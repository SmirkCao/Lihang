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

        import matplotlib.pyplot as plt 
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

        import matplotlib.pyplot as plt
        count, bins, ignored = plt.hist(s, 300, density=True)
        # draw PDF
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp(-(bins - mu)**2 / (2 * sigma**2)),
                 linewidth=2, color='r')
        plt.show()

    def test_alg_1901(self):
        pass
