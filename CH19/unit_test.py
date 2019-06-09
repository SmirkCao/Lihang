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

    def test_alg_1901(self):
        print("\n", np.__version__)
        # 以下，1.16 及以下， numpy 1.17的接口变了
        s = np.random.uniform(-1, 0, 1000)
        print(s.shape)
    