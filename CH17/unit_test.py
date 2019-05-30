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
import matplotlib.pyplot as plt
import numpy as np


class TestLSAMethods(unittest.TestCase):
    def test_lsa_puffinwarellc_tutorial(self):
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
        u, s, vh = np.linalg.svd(x, )
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

        u, s, vh = np.linalg.svd(x, )
        print("\n")
        print(40*"*"+"u"+40*"*")
        print(np.round(u[:, :3], 2))
        print(40*"*"+"s"+40*"*")
        print(np.round(s[:3], 2))
        print(40*"*"+"vh"+40*"*")
        print(np.round(vh[:3, :], 2))

        print(40*"*"+"svh"+40*"*")
        print(np.round(np.dot(s[:3]*np.eye(3), vh[:3, :]), 2))
