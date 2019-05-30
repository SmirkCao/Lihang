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
